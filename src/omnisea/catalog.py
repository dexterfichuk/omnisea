"""The :class:`Catalog` — what :func:`omnisea.discover` returns, and the gate before a download.

The research doc separates discovery (step 3) from retrieval (step 4) for a good reason: marine
queries can quietly become enormous, and the moment to notice is *before* the request goes out.
A Catalog is a cheap, printable list of what is available, with a ``.fetch()`` on the end of it.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import pandas as pd
import xarray as xr

from .errors import OmniseaError, PayloadTooLargeError, ProviderError
from .http import DEFAULT_MAX_WORKERS, map_threads
from .providers.base import DataSource, StationMatch, StationSeries
from .query import Query
from .registry import get_source
from .tree import build_tree, data_nodes

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
    "server",
    "status",
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

    #: The columns worth reading at a glance. `.frame` keeps all twelve.
    DISPLAY_COLUMNS = ["source", "station_id", "name", "distance_km", "n_rows_est", "variables"]

    def describe(self, width: int = 34) -> pd.DataFrame:
        """A printable view: the columns that help you decide, with long cells elided.

        Printing the full frame produced 725-character lines — the `variables` column dumps
        every CF name a source can serve — which destroys a terminal and hides the numbers the
        decision actually turns on. :attr:`frame` is still the complete table.

        This is what the catalogue's own repr shows, available on its own so a report or a
        notebook can print a filtered view without reaching for a private method.
        """
        frame = self.frame
        columns = [c for c in self.DISPLAY_COLUMNS if c in frame.columns]
        if self.query.sites and "site" in frame.columns and len(self.query.sites) > 1:
            columns.insert(1, "site")
        out = frame[columns].copy()
        for column in out.columns:
            if out[column].dtype == object or pd.api.types.is_string_dtype(out[column]):
                out[column] = out[column].map(
                    lambda v: v if not isinstance(v, str) or len(v) <= width
                    else v[: width - 1] + "\u2026"
                )
        if "distance_km" in out.columns:
            # None for a pure bbox query — there is no site to be near — so round elementwise.
            out["distance_km"] = out["distance_km"].map(
                lambda v: round(v, 2) if isinstance(v, (int, float)) else v
            )
        return out

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
        # The same columns whether or not sites were requested — an area query returning a
        # differently-shaped frame turned coverage()["has_match"] into a KeyError depending on
        # how the caller had asked for their data.
        if not self.query.sites:
            return pd.DataFrame(
                columns=["site", "n_stations", "n_rows_est", "sources", "has_match"]
            )
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
        **options: Any,
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

        if on_error == "raise" and self.errors:
            # The documented default. A one-shot omnisea.fetch() runs discovery internally and
            # never shows the Catalog, so a source that died there produced an empty tree, exit
            # code 0, and a nightly job publishing nothing while claiming success. Discovery
            # still *collects* — one dead API must not sink a survey — but turning that
            # collection into a silent success at fetch time is the failure this library exists
            # to prevent. Pass on_error="collect" to proceed with what did answer.
            failed = "; ".join(
                f"{name}: {message}" for name, message in sorted(self.errors.items())
            )
            raise ProviderError(
                f"{len(self.errors)} source(s) failed during discovery, so this result would be "
                f"incomplete: {failed}\n"
                "  Pass on_error='collect' to fetch what did answer; the failures are then "
                "recorded in the tree's omnisea_fetch_errors attribute."
            )
        ceiling = max_rows if max_rows is not None else self.query.max_rows
        # Lazy griddap matches are excluded from the pre-fetch ceiling: opening one downloads
        # nothing, and their catalogue estimate is the honest FULL-grid cell count — charging
        # a 17-million-cell model grid against a 2-million row ceiling would refuse a fetch
        # that costs a metadata read. The grid's own guards are omnisea_cells_estimate on the
        # node, align() refusing to flatten it, and to_netcdf() warning before materializing.
        estimate = int(sum(
            m.n_rows_est for m in self.matches
            if (m.extra or {}).get("protocol") != "griddap"
        ))
        if ceiling and estimate > ceiling:
            # The suggested max_rows is underscore-separated so it can be pasted straight into
            # Python. Only that number: applying .replace() to the whole message turned every
            # comma in the prose into an underscore.
            suggestion = f"{estimate + 1:,}".replace(",", "_")
            raise PayloadTooLargeError(
                f"this catalogue would pull about {estimate:,} rows across "
                f"{len(self.matches)} station(s), over the {ceiling:,} row ceiling.\n"
                "  Narrow it with .filter(nearest=1) or .filter(max_distance_km=...), "
                "shorten the time window, coarsen resolution (e.g. resolution='SIXTY_MINUTES'), "
                f"or raise the ceiling with fetch(max_rows={suggestion}).",
                estimate=estimate,
                limit=ceiling,
            )

        query = self.query
        if options:
            # The same provider options omnisea.fetch() takes -- series=, resolution=,
            # onc_token= -- so the documented discover -> filter -> fetch workflow can carry
            # one instead of forcing it back onto discover(). Validated the same way, so a
            # typo still gets the did-you-mean error rather than a silent no-op.
            query = query.replace(options={**dict(query.options), **options})
        if to_cf_units:
            query = query.replace(options={**dict(query.options), "to_cf_units": True})
        # Everything that shaped this result, so a saved tree can be re-run. Without these you
        # cannot tell a two-station tree produced by nearest=2 from one where only two stations
        # existed — and the README claims "the root records the query".
        retrieval = {
            "omnisea_fetched_stations": len(self.matches),
            "omnisea_group_by_site": int(group_by_site),
            "omnisea_max_rows": int(ceiling) if ceiling else 0,
            "omnisea_on_error": on_error,
        }

        by_source: dict[str, list[StationMatch]] = {}
        for m in self.matches:
            source = get_source(m.source)
            if source.discovery_only:
                # Metadata catalogues describe data; they have no arrays to contribute.
                continue
            by_source.setdefault(m.source, []).append(m)

        if not by_source:
            tree = self._record_incompleteness(
                build_tree(query, [], group_by_site=group_by_site), {}, retrieval
            )
            if not self.errors and not self.notes:
                # Nothing failed and nothing was out of range — the query simply matched no
                # station. discover()'s repr says so helpfully; a one-shot fetch() showed an
                # empty tree with no explanation at all, and the user cannot tell that from a
                # server being down or from having called it wrong.
                tree.attrs["omnisea_empty_reason"] = (
                    "No station matched this query. Nothing failed — the area, time window and "
                    "variable filter simply had no overlap. Try a larger radius_km, a wider "
                    "time window, omnisea.sources() to see what is registered, or "
                    "omnisea.discover(...) to inspect the catalogue before fetching."
                )
                log.info("fetch() matched no stations: %s", tree.attrs["omnisea_empty_reason"])
            return tree

        failures: dict[str, str] = {}

        def _run(item: tuple[str, list[StationMatch]]) -> list[StationSeries | xr.Dataset]:
            name, matches = item
            source: DataSource = get_source(name)
            log.debug("fetching %d station(s) from %s", len(matches), name)
            try:
                produced = source.fetch(query, matches)
                return _validated(name, produced)
            except OmniseaError as exc:
                # Already one of ours: propagate or collect, but never re-wrap.
                if on_error == "raise":
                    raise
                failures[name] = f"{type(exc).__name__}: {exc}"
                log.warning("fetch failed for %s: %s", name, exc)
                return []
            except Exception as exc:  # noqa: BLE001 - re-raised below unless collecting
                if on_error == "raise":
                    # "except omnisea.OmniseaError" is documented as a complete catch, and that
                    # promise is only as strong as the least careful installed plugin. Wrap
                    # anything a source lets escape so the guarantee actually holds.
                    raise ProviderError(
                        f"{type(exc).__name__} while fetching: {exc}", provider=name
                    ) from exc
                failures[name] = f"{type(exc).__name__}: {exc}"
                log.warning("fetch failed for %s: %s", name, exc)
                return []

        def _validated(name: str, produced: Any) -> list[StationSeries | xr.Dataset]:
            """Check what a source returned, naming the source rather than leaking an attribute.

            A wrong return type used to surface as ``AttributeError: 'str' object has no
            attribute 'is_empty'`` from deep in tree assembly — no source name, no expected
            type, and a private attribute as the only clue.
            """
            if produced is None:
                return []
            if isinstance(produced, (StationSeries, xr.Dataset)):
                raise ProviderError(
                    f"fetch() returned a bare {type(produced).__name__}; it must return a "
                    "*list* of StationSeries or xarray.Dataset. Wrap it: [series]",
                    provider=name,
                )
            if not isinstance(produced, list):
                raise ProviderError(
                    f"fetch() returned {type(produced).__name__}; expected "
                    "list[StationSeries | xarray.Dataset]",
                    provider=name,
                )
            for item in produced:
                if not isinstance(item, (StationSeries, xr.Dataset)):
                    raise ProviderError(
                        f"fetch() returned a list containing {type(item).__name__}; every "
                        "element must be a StationSeries or an xarray.Dataset",
                        provider=name,
                    )
            return produced

        results: list[StationSeries | xr.Dataset] = []
        for chunk in map_threads(
            _run, list(by_source.items()), max_workers=max_workers, label="source fetch"
        ):
            results.extend(chunk)

        tree = self._record_incompleteness(
            build_tree(query, results, group_by_site=group_by_site), failures, retrieval
        )
        if to_cf_units:
            _warn_unconverted(tree)
        return tree

    def _record_incompleteness(
        self, tree: xr.DataTree, failures: Mapping[str, str], retrieval: Mapping[str, Any] = {},
    ) -> xr.DataTree:
        """Stamp what went wrong onto the tree, from *both* steps of the retrieval.

        Discovery failures used to stop at the :class:`Catalog`, which is fine when someone
        prints it and decides — but a one-shot :func:`omnisea.fetch` never shows that object,
        so a source that died during discovery produced a tree with no trace of it. An empty
        tree then reads as "there is nothing here" when the truth was "a server was down",
        which is exactly the silent wrongness this library exists to prevent.

        Recorded rather than raised: :func:`omnisea.discover` deliberately survives one dead
        API so the other sixteen still answer, and a caller who printed the catalogue could
        already see the failure and chose to continue. What must not happen is that it
        disappears — so it lands here, and :func:`omnisea.citation` reports it in the
        attribution block a result gets published with.
        """
        tree.attrs.update(dict(retrieval))
        problems = {f"{name} (discovery)": message for name, message in self.errors.items()}
        problems.update({f"{name} (fetch)": message for name, message in failures.items()})
        if problems:
            tree.attrs["omnisea_fetch_errors"] = "; ".join(
                f"{name}: {message}" for name, message in sorted(problems.items())
            )
            tree.attrs["omnisea_fetch_incomplete"] = 1
        if self.notes:
            # Not failures — a rolling archive that cannot reach the requested dates, say.
            # Still the difference between "no station here" and "wrong collection for 2019".
            tree.attrs["omnisea_source_notes"] = "; ".join(
                f"{name}: {message}" for name, message in sorted(self.notes.items())
            )
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
        hidden = [c for c in COLUMNS if c not in self.describe().columns]
        footer_columns = (
            f"\n\n  (showing {len(self.describe().columns)} of {len(COLUMNS)} columns; "
            f"catalog.frame has them all: {', '.join(hidden)})"
        )
        body = self.describe().to_string(max_rows=25, index=False)
        footer = ""
        missing = self.missing_sites
        if missing:
            shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
            footer = f"\n\n  {len(missing)} site(s) with no match: {shown}"
        for name, message in self.notes.items():
            footer += f"\n  - {name}: {message}"
        for name, message in self.errors.items():
            footer += f"\n  ! {name} failed during discovery: {message}"
        return f"{header}\n{body}{footer_columns}{footer}"

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
            f"{self.describe(width=60).to_html(max_rows=40, index=False)}{note}"
        )


def _nearest_per_site(matches: list[StationMatch], n: int) -> list[StationMatch]:
    """Keep the ``n`` closest stations for each (site, source) pair.

    ``distance_km`` is ``None`` for a pure bbox query — there is no site to be near — so the
    stations sort equal and "nearest" degenerates to "first n discovered". That is a real
    answer to a question that has no better one, but it is not what the name promises, so it
    is worth knowing rather than guessing at.
    """
    if n < 1:
        raise ValueError(f"nearest must be at least 1; got {n}")
    if matches and all(m.distance_km is None for m in matches):
        log.warning(
            "filter(nearest=%d) on a query with no sites: these matches carry no distance, so "
            "this keeps the first %d per source in discovery order rather than the closest. "
            "Use sites=/lat=/lon= if you meant proximity.",
            n,
            n,
        )
    buckets: dict[tuple[str | None, str], list[StationMatch]] = {}
    for m in matches:
        buckets.setdefault((m.site, m.source), []).append(m)
    kept: list[StationMatch] = []
    for group in buckets.values():
        # Distance first, then the longer record. ECCC splits one physical site across station
        # ids — TOFINO A is 1038204 and 1038210 at identical coordinates — and a pure distance
        # sort picked between them arbitrarily, silently costing a user 46% of their
        # temperature series with no signal that a co-located alternative existed.
        group.sort(
            key=lambda m: (
                m.distance_km if m.distance_km is not None else 0.0,
                -int(m.n_rows_est or 0),
                str(m.station_id),
            )
        )
        kept.extend(group[:n])
        for dropped in group[n:]:
            twin = group[n - 1]
            if dropped.distance_km == twin.distance_km and dropped.name == twin.name:
                log.info(
                    "nearest=%d kept %s and dropped %s — same name and position (%s). One "
                    "physical site split across station ids; the other may hold a longer "
                    "record. Use .filter(station_id=...) to choose.",
                    n, twin.station_id, dropped.station_id, twin.name,
                )
    return kept


def _warn_unconverted(tree: xr.DataTree) -> None:
    """Say which sources ``to_cf_units=True`` could not convert, because it is not all of them.

    The switch converts wherever a field table states the canonical units to convert *to*.
    ERDDAP states the units its numbers are in and nothing about reaching CF ones, so omnisea
    will not guess a scale factor for someone else's data — which is right, and silent. A
    cross-border frame then came back with Canadian air temperature in kelvin beside American
    air temperature in degrees Celsius, under one column name each, from the switch whose whole
    promise is one uniform unit system. It has to say so.
    """
    stranded: dict[str, set[str]] = {}
    converted: set[str] = set()
    for _path, ds in data_nodes(tree):
        source = str(ds.attrs.get("source_name") or "?")
        for name, variable in ds.data_vars.items():
            attrs = variable.attrs
            if str(name).endswith("_qc"):
                continue
            if attrs.get("omnisea_converted_from"):
                converted.add(source)
            elif str(attrs.get("units") or ""):
                stranded.setdefault(source, set()).add(str(name))
    # A source that converted *something* has a field table stating CF units where they differ;
    # what it left alone is already canonical (percent, degree, a dimensionless count). The
    # asymmetry worth warning about is a source that could convert nothing at all.
    stranded = {s: names for s, names in stranded.items() if s not in converted}
    if not stranded:
        return
    detail = "; ".join(
        f"{source} ({len(names)} variable(s), e.g. {sorted(names)[0]})"
        for source, names in sorted(stranded.items())
    )
    warnings.warn(
        f"to_cf_units=True converted nothing from: {detail}. Those sources publish the units "
        "their numbers are in but not how to reach canonical CF units, and omnisea will not "
        "guess a scale factor for someone else's data — so this result mixes unit systems. "
        'Every column\'s units are in the node attrs and in align()\'s '
        'attrs["omnisea_units"]; check them before comparing columns across sources.',
        UserWarning,
        stacklevel=3,
    )
