"""An executable version of the provider contract.

``docs/adding-a-provider.md`` describes what a source must do. Prose is easy to skim past, so
the same rules live here as checks a contributor can run before opening a pull request, and CI
runs over every registered source::

    python -m omnisea.conformance          # check everything registered
    omnisea.check_source(MySource(MyProvider()))

The checks are the ones where being wrong is *quiet*: a standard name that does not exist, an
aggregate with no ``cell_methods`` for :func:`omnisea.align` to read, a variable name that
collides with a QC column. None of those raise at runtime; they just make results subtly wrong.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["Problem", "check_source", "check_all", "cf_standard_names", "format_report"]

_CF_NAMES_FILE = Path(__file__).parent / "data" / "cf-standard-names.txt.gz"

#: CF discrete-sampling-geometry types a source may declare.
FEATURE_TYPES = frozenset(
    {"point", "timeSeries", "trajectory", "profile", "timeSeriesProfile",
     "trajectoryProfile", "grid", "metadata"}
)

#: Prefixes providers use on *their own* field names to mark an interval statistic. Matched on
#: the raw field rather than the output variable: CF names legitimately contain "maximum" and
#: "mean" as part of a phrase (``air_pressure_at_mean_sea_level``,
#: ``sea_surface_wave_period_at_variance_spectral_density_maximum``), and flagging those would
#: train contributors to ignore the warning.
AGGREGATE_PREFIXES = (
    "avg_", "max_", "min_", "mean_", "total_", "sum_",
    "AVG_", "MAX_", "MIN_", "MEAN_", "TOTAL_", "SUM_", "SPEED_MAX_", "DIRECTION_MAX_",
)

#: Variables that are a *timestamp of* an extreme, not the extreme itself. A time has no
#: meaningful aggregation, which is why the sources that emit them leave cell_methods off.
TIME_VALUED_SUFFIXES = ("_time", "_date", "_tm")

_NETCDF_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Problem:
    """One contract violation. ``level`` is ``"error"`` or ``"warning"``."""

    level: str
    source: str
    message: str

    def __str__(self) -> str:
        mark = "ERROR  " if self.level == "error" else "warning"
        return f"{mark} {self.source}: {self.message}"


def cf_standard_names() -> frozenset[str]:
    """The CF standard name table, bundled so this works offline and in CI.

    Aliases are included: they are valid input, and a source emitting one is not wrong, only
    old-fashioned.
    """
    with gzip.open(_CF_NAMES_FILE, "rt", encoding="utf-8") as handle:
        return frozenset(
            line.strip()
            for line in handle
            if line.strip() and not line.startswith("#")
        )


def check_source(source: Any, *, cf_names: Iterable[str] | None = None) -> list[Problem]:
    """Check one :class:`~omnisea.DataSource` against the contract.

    Returns a list of :class:`Problem`; an empty list means it conforms.
    """
    valid = frozenset(cf_names) if cf_names is not None else cf_standard_names()
    name = getattr(source, "name", "") or type(source).__name__
    problems: list[Problem] = []

    def err(message: str) -> None:
        problems.append(Problem("error", name, message))

    def warn(message: str) -> None:
        problems.append(Problem("warning", name, message))

    # ---------------------------------------------------------------- identity
    if not getattr(source, "name", ""):
        err("has no .name, so it cannot be registered or selected")
    if not getattr(source, "node_path", ""):
        err("has no .node_path, so its nodes have nowhere to live in the tree")
    if not getattr(source, "title", ""):
        warn("has no .title; the Catalog and node attributes will read poorly")

    feature_type = getattr(source, "feature_type", "")
    if feature_type not in FEATURE_TYPES:
        err(
            f"feature_type {feature_type!r} is not a CF DSG type "
            f"({', '.join(sorted(FEATURE_TYPES))})"
        )

    provider = getattr(source, "provider", None)
    if provider is None:
        err("has no .provider, so its data cannot be attributed")
    else:
        if not getattr(provider, "license", ""):
            err(f"provider {provider.name!r} declares no license; data cannot be attributed")
        if not getattr(provider, "title", ""):
            warn(f"provider {provider.name!r} has no .title for attribution")

    # ---------------------------------------------------------------- declared windows
    period = getattr(source, "period", None)
    if period is not None:
        try:
            pd.Period("2024-01-01", freq=period)
        except Exception:  # noqa: BLE001 - the point is to report it, not raise
            err(f"period {period!r} is not a pandas Period alias (use 'D', 'M', 'Y')")

    retention = getattr(source, "retention", None)
    if retention is not None and not isinstance(retention, pd.Timedelta):
        err(f"retention must be a pd.Timedelta; got {type(retention).__name__}")

    # ---------------------------------------------------------------- field table
    fields = getattr(source, "fields", {}) or {}
    if not fields and not getattr(source, "discovery_only", False):
        warn("declares no fields; every variable will be carried through unmapped")

    equivalents = getattr(source, "equivalent_fields", ()) or ()

    def declared_equivalent(a: str, b: str) -> bool:
        return any({a, b} <= group for group in equivalents)

    seen_vars: dict[str, str] = {}
    for raw, spec in fields.items():
        where = f"field {raw!r}"

        if not getattr(spec, "var", ""):
            err(f"{where} has no .var (the output variable name)")
            continue
        if not _NETCDF_NAME.match(spec.var):
            err(f"{where} maps to {spec.var!r}, which is not a valid netCDF variable name")
        if spec.var.endswith("_qc"):
            err(f"{where} maps to {spec.var!r}, which collides with the QC flag naming")
        if spec.var in seen_vars and not declared_equivalent(raw, seen_vars[spec.var]):
            err(
                f"{where} and {seen_vars[spec.var]!r} both map to {spec.var!r}. If they are the "
                "same measurement under two spellings, say so in equivalent_fields; otherwise "
                "give them distinct .var names."
            )
        seen_vars[spec.var] = raw

        standard_name = getattr(spec, "standard_name", "")
        if standard_name and standard_name not in valid:
            err(
                f"{where}: standard_name {standard_name!r} is not in the CF standard name "
                "table. Use '' with a good long_name if no CF name exists — never invent one."
            )
        if not standard_name and not getattr(spec, "long_name", ""):
            warn(f"{where} has neither a standard_name nor a long_name, so it is undescribed")

        if getattr(spec, "cf_units", None) and not getattr(spec, "units", None):
            if spec.units is None:
                pass  # units come from the record, which is fine
        looks_aggregated = raw.startswith(AGGREGATE_PREFIXES)
        is_timestamp = spec.var.endswith(TIME_VALUED_SUFFIXES)
        if looks_aggregated and not is_timestamp and not getattr(spec, "cell_methods", None):
            warn(
                f"{where} looks like an interval statistic but declares no cell_methods; "
                "align() will treat it as instantaneous and may interpolate it"
            )
        scale = getattr(spec, "cf_scale", 1.0)
        offset = getattr(spec, "cf_offset", 0.0)
        if (scale != 1.0 or offset != 0.0) and not getattr(spec, "cf_units", None):
            err(f"{where} converts values but declares no cf_units to convert them to")

    return problems


def check_all(sources: Iterable[Any] | None = None) -> list[Problem]:
    """Check every registered source (or the ones given)."""
    if sources is None:
        from .registry import all_sources

        sources = all_sources()
    valid = cf_standard_names()
    problems: list[Problem] = []
    for source in sources:
        problems.extend(check_source(source, cf_names=valid))
    return problems


def format_report(problems: list[Problem]) -> str:
    if not problems:
        return "All sources conform."
    errors = [p for p in problems if p.level == "error"]
    lines = [str(p) for p in problems]
    lines.append("")
    lines.append(f"{len(errors)} error(s), {len(problems) - len(errors)} warning(s)")
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - console entry point
    problems = check_all()
    print(format_report(problems))
    return 1 if any(p.level == "error" for p in problems) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
