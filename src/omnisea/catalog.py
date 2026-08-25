"""The :class:`Catalog` — what :func:`omnisea.discover` returns, and the gate before a download.

The research doc separates discovery (step 3) from retrieval (step 4) for a good reason: marine
queries can quietly become enormous, and the moment to notice is *before* the request goes out.
A Catalog is a cheap, printable list of what is available, with a ``.fetch()`` on the end of it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import pandas as pd
import xarray as xr

from .errors import PayloadTooLargeError
from .http import DEFAULT_MAX_WORKERS, map_threads
from .providers.base import DataSource, StationMatch, StationSeries
from .query import Query
from .registry import get_source
from .tree import build_tree

__all__ = ["Catalog"]

log = logging.getLogger("omnisea.catalog")

COLUMNS = [
    "source",
    "provider",
    "site",
    "station_id",
    "name",
    "lat",
    "lon",
    "distance_km",
    "variables",
    "n_rows_est",
    "first",
    "last",
]


class Catalog:
    """A discovered set of stations, filterable and then fetchable.

    Printing one is the point — it shows what each provider found, how far away it is and how
    many rows it would pull, so the decision to download is an informed one.
    """

    def __init__(
        self,
        query: Query,
        matches: Sequence[StationMatch],
        errors: Mapping[str, str] | None = None,
        notes: Mapping[str, str] | None = None,
    ):
        self.query = query
        self.matches: list[StationMatch] = list(matches)
        #: Sources that failed during discovery, ``{source_name: message}``. Recorded rather
        #: than raised so one unreachable API cannot sink an otherwise good multi-source query.
        self.errors: dict[str, str] = dict(errors or {})
        #: Sources that could not answer for a reason worth explaining — a rolling archive that
        #: does not reach back far enough, say. Not failures, but "no results" alone would be
        #: read as "there is nothing here", which is a different and wrong conclusion.
        self.notes: dict[str, str] = dict(notes or {})

    # ------------------------------------------------------------------ views

    @property
    def frame(self) -> pd.DataFrame:
        """The catalogue as a DataFrame, one row per station."""
        if not self.matches:
            return pd.DataFrame(columns=COLUMNS)
        frame = pd.DataFrame([m.as_row() for m in self.matches])
        for col in COLUMNS:
            if col not in frame.columns:
                frame[col] = None
        frame = frame[COLUMNS]
        sort_cols = ["distance_km"] if frame["distance_km"].notna().any() else ["source", "name"]
        return frame.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    def to_dataframe(self) -> pd.DataFrame:
        return self.frame

    @property
    def n_rows_est(self) -> int:
        """Total rows this catalogue would pull if fetched."""
        return int(sum(m.n_rows_est for m in self.matches))

    @property
    def sources(self) -> list[str]:
        """Data source names present, e.g. ``["dfo_tides", "eccc_climate"]``."""
        return sorted({m.source for m in self.matches})

    @property
    def providers(self) -> list[str]:
        """Organizations present, e.g. ``["dfo", "eccc"]``."""
        return sorted({m.provider for m in self.matches if m.provider})

    @property
    def sites(self) -> list[str]:
        """Requested site labels, in the order they were given."""
        return [s.label for s in self.query.sites]

    @property
    def missing_sites(self) -> list[str]:
        """Requested sites that no provider could match.

        Surfaced as a first-class property because with a long list of locations, the ones that
        found nothing are the result you most need to see.
        """
        found = {m.site for m in self.matches if m.site}
        return [label for label in self.sites if label not in found]

    def coverage(self) -> pd.DataFrame:
        """Per-site station counts, with a row for every requested site including empty ones."""
        if not self.query.sites:
            return pd.DataFrame(columns=["site", "n_stations", "n_rows_est", "providers"])
        by_site: dict[str, list[StationMatch]] = {label: [] for label in self.sites}
        for m in self.matches:
            if m.site in by_site:
                by_site[m.site].append(m)
        return pd.DataFrame(
            [
                {
                    "site": label,
                    "n_stations": len(ms),
                    "n_rows_est": int(sum(m.n_rows_est for m in ms)),
                    "sources": ", ".join(sorted({m.source for m in ms})),
                    "has_match": bool(ms),
                }
                for label, ms in by_site.items()
            ]
        )

    def metadata(self) -> pd.DataFrame:
        """Rows contributed by discovery-only sources, with their descriptive fields.

        Metadata catalogues answer "what exists here?" rather than returning arrays, so their
        useful output is a table of titles, extents, licences and download URLs — not a tree.
        """
        rows: list[dict[str, Any]] = []
        for m in self.matches:
            record = m.extra.get("record")
            if not isinstance(record, dict):
                continue
            rows.append(
                {
                    "source": m.source,
                    "site": m.site,
                    "id": m.station_id,
                    "title": record.get("title", ""),
                    "organization": record.get("organization", ""),
                    "eov": ", ".join(record.get("eov") or []),
                    "variables": ", ".join(m.variables),
                    "start": m.first,
                    "end": m.last,
                    "bbox": record.get("bbox"),
                    "license": record.get("license", ""),
                    "urls": ", ".join(d["url"] for d in record.get("distribution") or []),
                    "abstract": (record.get("abstract") or "")[:300],
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ filtering

    def filter(
        self,
        *,
        source: str | Iterable[str] | None = None,
        provider: str | Iterable[str] | None = None,
        site: str | Iterable[str] | None = None,
        station_id: str | Iterable[str] | None = None,
        variables: str | Iterable[str] | None = None,
        max_distance_km: float | None = None,
        max_rows: int | None = None,
        name_contains: str | None = None,
        nearest: int | None = None,
        where: Callable[[StationMatch], bool] | None = None,
    ) -> Catalog:
        """Narrow the catalogue. Returns a new :class:`Catalog`; the original is untouched.

        ``source`` selects a dataset (``"dfo_tides"``); ``provider`` selects an organization
        (``"eccc"``, matching all four of its datasets).

        ``nearest=n`` keeps the ``n`` closest stations *per site*, which is usually what you want
        for a multi-location query — a global top-n would happily return five stations clustered
        around one site and none for the rest.
        """
        matches = list(self.matches)

        def _as_set(value: Any) -> set[str] | None:
            if value is None:
                return None
            if isinstance(value, str):
                return {value}
            return {str(v) for v in value}

        sources = _as_set(source)
        providers = _as_set(provider)
        sites = _as_set(site)
        ids = _as_set(station_id)
        wanted_vars = _as_set(variables)

        if sources is not None:
            matches = [m for m in matches if m.source in sources]
        if providers is not None:
            matches = [m for m in matches if m.provider in providers]
        if sites is not None:
            matches = [m for m in matches if m.site in sites]
        if ids is not None:
            matches = [m for m in matches if str(m.station_id) in ids]
        if wanted_vars is not None:
            matches = [m for m in matches if wanted_vars & set(m.variables)]
        if max_distance_km is not None:
            matches = [
                m for m in matches if m.distance_km is None or m.distance_km <= max_distance_km
            ]
        if max_rows is not None:
            matches = [m for m in matches if m.n_rows_est <= max_rows]
        if name_contains is not None:
            needle = name_contains.lower()
            matches = [m for m in matches if needle in (m.name or "").lower()]
        if where is not None:
            matches = [m for m in matches if where(m)]

        if nearest is not None:
            matches = _nearest_per_site(matches, nearest)

        return Catalog(self.query, matches, self.errors, self.notes)

    # ------------------------------------------------------------------ retrieval

    def fetch(
        self,
        *,
        to_cf_units: bool = False,
        group_by_site: bool = False,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_rows: int | None = None,
        on_error: str = "raise",
    ) -> xr.DataTree:
        """Download the catalogued stations and assemble them into a tree.

        The row-count ceiling is checked here, before any bulk request is made, so an
        accidentally enormous query fails with an estimate and the knob to change rather than
        hammering the upstream API.

        ``on_error`` decides what a failing source does. The default ``"raise"`` is deliberately
        stricter than :func:`omnisea.discover`, which collects failures and carries on. The two
        steps answer different questions: discovery is a *survey*, and a source missing from it
        costs you options you can see are missing on the catalogue. A fetch produces the data
        you will actually analyse, and a tree quietly missing a source looks exactly like a tree
        where that source had nothing to say.

        Pass ``on_error="collect"`` for exploratory work where partial results are useful. The
        failures are then recorded in the tree's ``omnisea_fetch_errors`` attribute and logged
        as warnings, never dropped in silence.
        """
        if on_error not in ("raise", "collect"):
            raise ValueError(f"on_error must be 'raise' or 'collect'; got {on_error!r}")
        ceiling = max_rows if max_rows is not None else self.query.max_rows
        estimate = self.n_rows_est
        if ceiling and estimate > ceiling:
            raise PayloadTooLargeError(
                f"this catalogue would pull about {estimate:,} rows across "
                f"{len(self.matches)} station(s), over the {ceiling:,} row ceiling.\n"
                "  Narrow it with .filter(nearest=1) or .filter(max_distance_km=...), "
                "shorten the time window, coarsen resolution (e.g. resolution='SIXTY_MINUTES'), "
                f"or raise the ceiling with fetch(max_rows={estimate + 1:,}).".replace(",", "_"),
                estimate=estimate,
                limit=ceiling,
            )

        query = self.query
        if to_cf_units:
            query = query.replace(options={**dict(query.options), "to_cf_units": True})

        by_source: dict[str, list[StationMatch]] = {}
        for m in self.matches:
            source = get_source(m.source)
            if source.discovery_only:
                # Metadata catalogues describe data; they have no arrays to contribute.
                continue
            by_source.setdefault(m.source, []).append(m)

        if not by_source:
            return build_tree(query, [], group_by_site=group_by_site)

        failures: dict[str, str] = {}

        def _run(item: tuple[str, list[StationMatch]]) -> list[StationSeries | xr.Dataset]:
            name, matches = item
            source: DataSource = get_source(name)
            log.debug("fetching %d station(s) from %s", len(matches), name)
            try:
                return source.fetch(query, matches)
            except Exception as exc:  # noqa: BLE001 - re-raised below unless collecting
                if on_error == "raise":
                    raise
                failures[name] = f"{type(exc).__name__}: {exc}"
                log.warning("fetch failed for %s: %s", name, exc)
                return []

        results: list[StationSeries | xr.Dataset] = []
        for chunk in map_threads(
            _run, list(by_source.items()), max_workers=max_workers, label="source fetch"
        ):
            results.extend(chunk)

        tree = build_tree(query, results, group_by_site=group_by_site)
        if failures:
            tree.attrs["omnisea_fetch_errors"] = "; ".join(
                f"{name}: {message}" for name, message in sorted(failures.items())
            )
            tree.attrs["omnisea_fetch_incomplete"] = 1
        return tree

    # ------------------------------------------------------------------ dunders

    def __len__(self) -> int:
        return len(self.matches)

    def __bool__(self) -> bool:
        return bool(self.matches)

    def __iter__(self) -> Iterator[StationMatch]:
        return iter(self.matches)

    def __getitem__(self, index: int) -> StationMatch:
        return self.matches[index]

    def __repr__(self) -> str:  # pragma: no cover - display only
        if not self.matches:
            hint = ""
            if self.query.sites:
                hint = f" for {len(self.query.sites)} site(s)"
            lines = [f"<Catalog: no stations found{hint} in {self.query!r}>"]
            for name, message in self.notes.items():
                lines.append(f"  - {name}: {message}")
            for name, message in self.errors.items():
                lines.append(f"  ! {name} failed during discovery: {message}")
            if not self.notes and not self.errors:
                lines.append(
                    "  Try a larger radius_km, a wider time window, or omnisea.sources() to "
                    "see what is registered."
                )
            return "\n".join(lines)
        header = (
            f"<Catalog: {len(self.matches)} station(s) from {len(self.sources)} source(s), "
            f"~{self.n_rows_est:,} rows>"
        )
        body = self.frame.to_string(max_rows=40, index=False)
        footer = ""
        missing = self.missing_sites
        if missing:
            shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
            footer = f"\n\n  {len(missing)} site(s) with no match: {shown}"
        for name, message in self.notes.items():
            footer += f"\n  - {name}: {message}"
        for name, message in self.errors.items():
            footer += f"\n  ! {name} failed during discovery: {message}"
        return f"{header}\n{body}{footer}"

    def _repr_html_(self) -> str:  # pragma: no cover - notebook display only
        if not self.matches:
            return f"<pre>{self!r}</pre>"
        missing = self.missing_sites
        note = (
            f"<p><em>{len(missing)} site(s) with no match: "
            f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}</em></p>"
            if missing
            else ""
        )
        return (
            f"<p><strong>Catalog</strong>: {len(self.matches)} station(s) from "
            f"{len(self.sources)} source(s), ~{self.n_rows_est:,} rows estimated</p>"
            f"{self.frame.to_html(max_rows=40, index=False)}{note}"
        )


def _nearest_per_site(matches: list[StationMatch], n: int) -> list[StationMatch]:
    """Keep the ``n`` closest stations for each (site, source) pair."""
    buckets: dict[tuple[str | None, str], list[StationMatch]] = {}
    for m in matches:
        buckets.setdefault((m.site, m.source), []).append(m)
    kept: list[StationMatch] = []
    for group in buckets.values():
        group.sort(key=lambda m: (m.distance_km if m.distance_km is not None else 0.0))
        kept.extend(group[:n])
    return kept
