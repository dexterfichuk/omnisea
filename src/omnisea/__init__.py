"""omnisea — a unified Python client for ocean, tidal, and weather data.

Marine data access is fragmented across provider-specific APIs. omnisea is the middleware layer
that sits above them: pluggable adapters per source, CF-standard canonicalization of names and
units, EDR-shaped spatial-temporal queries, and :class:`xarray.DataTree` as the output container
so that 1-D point series and 4-D grids can coexist without a lossy flat join.

    >>> import omnisea
    >>> cat = omnisea.discover(bbox=(-124, 48, -123, 49), time=("2024-07-01", "2024-07-08"))
    >>> print(cat)                     # look before you download
    >>> tree = cat.fetch()             # then pull the confirmed subset
    >>> print(omnisea.summary(tree))

Two things omnisea deliberately does *not* do: it does not silently rescale values into
canonical units (pass ``to_cf_units=True`` if you want that), and it does not drop fields it has
no CF mapping for (they travel under the provider's own names).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd
import xarray as xr

__version__ = "0.1.0"

from . import cf, registry
from .align import (
    add_local,
    aggregation_for,
    align,
    correlations,
    drop_correlated,
    model_matrix,
)
from .catalog import Catalog
from .errors import (
    MissingDependencyError,
    OmniseaError,
    PayloadTooLargeError,
    ProviderError,
    QueryError,
    UnknownProviderError,
    UpstreamError,
)
from .http import (
    DEFAULT_MAX_WORKERS,
    clear_caches,
    disable_cache,
    enable_cache,
    map_threads,
    set_max_concurrency,
    set_timeout,
)
from .provenance import citation, provenance, sources_used
from .providers import BUILTIN_PROVIDERS
from .providers.base import (
    DataSource,
    DiscoverySource,
    Provider,
    RetrievalSource,
    StationMatch,
    StationSeries,
)
from .query import BBox, Query, Site, as_sites, register_option
from .registry import register_provider, register_source
from .tree import (
    build_tree,
    coverage,
    describe,
    fields,
    query_attrs,
    stations,
    summary,
    to_dataframe,
    to_netcdf,
)

log = logging.getLogger("omnisea")

__all__ = [
    "__version__",
    # modules re-exported for convenience
    "cf",
    "registry",
    "build_tree",
    "as_sites",
    # queries
    "discover",
    "fetch",
    "position",
    "positions",
    "area",
    "Query",
    "Site",
    "BBox",
    "Catalog",
    # introspection
    "providers",
    "sources",
    "variables",
    "erddap_servers",
    # alignment / modelling
    "align",
    "add_local",
    "aggregation_for",
    "correlations",
    "drop_correlated",
    "model_matrix",
    # tree helpers
    "summary",
    "describe",
    "fields",
    "provenance",
    "citation",
    "sources_used",
    "stations",
    "to_dataframe",
    "to_netcdf",
    "coverage",
    "query_attrs",
    # extension
    "Provider",
    "DataSource",
    "RetrievalSource",
    "DiscoverySource",
    "StationMatch",
    "StationSeries",
    "register_provider",
    "register_source",
    "register_option",
    "check_source",
    "check_all",
    "set_max_concurrency",
    "set_timeout",
    "enable_cache",
    "disable_cache",
    "clear_caches",
    # errors
    "OmniseaError",
    "QueryError",
    "ProviderError",
    "UpstreamError",
    "UnknownProviderError",
    "PayloadTooLargeError",
    "MissingDependencyError",
]

for _provider in BUILTIN_PROVIDERS:
    registry.register_provider(_provider)


def __dir__() -> list[str]:
    """Only the public API. Without this, `dir(omnisea)` and tab-completion also offered
    `Any`, `Iterable`, `Sequence`, `pd`, `xr`, `logging` and `annotations`."""
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    # The conformance checker is loaded on first use rather than at import, so that its
    # documented invocation — ``python -m omnisea.conformance`` — does not trip runpy's
    # "found in sys.modules" warning by having been pre-imported here.
    if name in ("check_source", "check_all"):
        from . import conformance

        return getattr(conformance, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------- introspection


def providers() -> list[str]:
    """Registered organizations, e.g. ``["cioos", "dfo", "eccc"]``.

    Pass any of these to ``providers=`` to select every dataset that organization publishes.
    """
    return registry.provider_names()


def sources() -> list[str]:
    """Registered datasets, e.g. ``["cioos_metadata", "dfo_tides", "eccc_climate", ...]``.

    These are the individual adapters; ``providers=`` accepts these names too.
    """
    return registry.source_names()


def erddap_servers() -> pd.DataFrame:
    """ERDDAP installations omnisea knows by name, and what each one is worth querying for.

    ERDDAP is the same software at a few hundred institutions and omnisea reads any of them, so
    the barrier is not technical — it is that you cannot query a server you have never heard of.
    Pass a name from the ``server`` column to ``erddap_server=``, a list of them, or ``"all"``
    to sweep every one::

        omnisea.discover(sites=[...], time=(...), providers="erddap",
                         erddap_server=["cioos_pacific", "hakai", "salishseacast"])

    The table is a convenience, not a boundary: ``erddap_server=`` still takes any URL, and an
    installation that is not listed is not unsupported.
    """
    from .providers.erddap.servers import server_table

    return pd.DataFrame(server_table())


def variables() -> pd.DataFrame:
    """CF standard names omnisea can serve, and which sources serve each one.

    This is a **floor, not an inventory**. It lists the names omnisea curates; every source also
    returns whatever else the platform published, under the provider's own field names. SWOB
    alone ships about 74 fields of which 12 are named here. To see what a particular fetch
    really returned, use :func:`fields`.
    """
    rows: list[dict[str, Any]] = []
    for source in registry.all_sources():
        for raw, spec in source.fields.items():
            rows.append(
                {
                    "variable": spec.var,
                    "standard_name": spec.standard_name,
                    "units": spec.units or "(from data)",
                    "cf_units": spec.cf_units or spec.units or "",
                    "source": source.name,
                    "provider": source.provider.name,
                    "raw_field": raw,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["variable", "source"]).reset_index(drop=True)
    return _VariablesFrame(frame)


class _VariablesFrame(pd.DataFrame):
    """The :func:`variables` table, with membership testing that means what it looks like.

    ``"air_temperature" in omnisea.variables()`` reads as a question about variables, but a
    plain DataFrame answers it about *column* names — so it returned False for every real
    variable and True for "units". Containment here asks the question the reader meant.
    """

    _metadata: list[str] = []

    @property
    def _constructor(self):
        return _VariablesFrame

    def __contains__(self, key: object) -> bool:
        name = str(key)
        if name in self.columns:
            return True
        return bool(
            (self["variable"] == name).any() or (self["standard_name"] == name).any()
        )

    def names(self) -> list[str]:
        """Just the variable names, for when a list is what you wanted."""
        return sorted(set(self["variable"]) | (set(self["standard_name"]) - {""}))


# --------------------------------------------------------------------------- queries


def _build_query(
    *,
    bbox: Sequence[float] | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 25.0,
    sites: Any = None,
    time: Any = None,
    variables: Iterable[str] | None = None,
    depth: Sequence[float] | None = None,
    providers: Iterable[str] | str | None = None,
    max_rows: int = 2_000_000,
    **options: Any,
) -> Query:
    if time is None:
        raise QueryError("time is required, e.g. time=('2024-07-01', '2024-07-08')")

    # Options are validated first: `discover(latitude=..., longitude=...)` otherwise dies on
    # "give one of bbox=, sites= or lat/lon=" without ever mentioning the keys actually typed,
    # while the option validator would have said "did you mean 'lat'?".
    from .query import _validate_options

    _validate_options(options)

    given = [k for k, v in (("bbox", bbox), ("sites", sites), ("lat", lat)) if v is not None]
    if len(given) > 1:
        raise QueryError(f"give only one of bbox=, sites= or lat/lon=; got {', '.join(given)}")

    if isinstance(providers, str):
        providers = [providers]

    if sites is not None:
        return Query.from_sites(
            sites,
            time,
            radius_km=radius_km,
            variables=variables,
            depth=depth,
            providers=providers,
            max_rows=max_rows,
            **options,
        )
    if lat is not None:
        if lon is None:
            raise QueryError("lat given without lon")
        return Query.from_position(
            lat,
            lon,
            time,
            radius_km=radius_km,
            variables=variables,
            depth=depth,
            providers=providers,
            max_rows=max_rows,
            **options,
        )
    if bbox is not None:
        return Query.from_area(
            bbox,
            time,
            variables=variables,
            depth=depth,
            providers=providers,
            max_rows=max_rows,
            **options,
        )
    raise QueryError("give one of bbox=, sites= or lat/lon=")


def discover(
    *,
    bbox: Sequence[float] | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 25.0,
    sites: Any = None,
    time: Any = None,
    variables: Iterable[str] | None = None,
    depth: Sequence[float] | None = None,
    providers: Iterable[str] | str | None = None,
    max_rows: int = 2_000_000,
    max_workers: int = DEFAULT_MAX_WORKERS,
    **options: Any,
) -> Catalog:
    """Find out what data exists for a query, without downloading any of it.

    ``time`` accepts a ``(start, end)`` pair, a ``slice``, or a single date meaning that whole
    day — so ``time="2024-07-01"`` is 1 July, and ``time="2024"`` is **1 January 2024 only**,
    not the year. Write a pair for a range: ``time=("2024-01-01", "2025-01-01")``. Naive
    timestamps are read as UTC, which is what every marine API speaks.


    Returns a :class:`Catalog` — a printable table of stations with row estimates — which you
    then narrow with ``.filter(...)`` and pull with ``.fetch()``. Separating the two steps is
    what stops an innocent-looking bbox from turning into a multi-million-row download.

    A source that fails is recorded on the catalogue rather than aborting the others; check
    ``catalog.errors``.
    """
    query = _build_query(
        bbox=bbox,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        sites=sites,
        time=time,
        variables=variables,
        depth=depth,
        providers=providers,
        max_rows=max_rows,
        **options,
    )
    return discover_query(query, max_workers=max_workers)


def discover_query(query: Query, *, max_workers: int = DEFAULT_MAX_WORKERS) -> Catalog:
    """Run discovery for an already-built :class:`Query`."""
    selected = registry.select(query.providers)
    errors: dict[str, str] = {}
    notes: dict[str, str] = {}

    # Rolling archives are checked before we call them. A source that cannot reach back to the
    # requested dates would otherwise return nothing, and "no results" reads as "there is no
    # station here" — a different and wrong conclusion from "this collection only keeps 30 days".
    runnable: list[DataSource] = []
    for source in selected:
        gap = source.retention_gap(query)
        if gap:
            notes[source.name] = gap
        if source.covers(query):
            runnable.append(source)

    def _discover(source: DataSource) -> list[StationMatch]:
        try:
            found = source.discover(query)
        except Exception as exc:  # noqa: BLE001 - one dead API must not sink the rest
            errors[source.name] = f"{type(exc).__name__}: {exc}"
            log.warning("discovery failed for %s: %s", source.name, exc)
            return []
        # Read in this thread, immediately: a source that answered only partly says so here,
        # and a partial answer returned in silence reads as a complete one.
        note = source.take_discovery_note()
        if note:
            notes[source.name] = (
                f"{notes[source.name]}; {note}" if source.name in notes else note
            )
        return found

    matches: list[StationMatch] = []
    # One thread per source, up to a sane cap: discovery is I/O-bound waiting on ~26
    # independent institutions, and rationing it to 8 threads made the wall time the sum of
    # the slowest rounds instead of the single slowest server. The global request semaphore
    # still bounds true concurrency per process.
    fan_out = max(max_workers, min(32, len(runnable)))
    for found in map_threads(_discover, runnable, max_workers=fan_out, label="discovery"):
        matches.extend(found)

    return Catalog(query, matches, errors, notes)


def fetch(
    *,
    bbox: Sequence[float] | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 25.0,
    sites: Any = None,
    time: Any = None,
    variables: Iterable[str] | None = None,
    depth: Sequence[float] | None = None,
    providers: Iterable[str] | str | None = None,
    max_rows: int = 2_000_000,
    max_workers: int = DEFAULT_MAX_WORKERS,
    to_cf_units: bool = False,
    group_by_site: bool = False,
    nearest: int | None = None,
    on_error: str = "raise",
    **options: Any,
) -> xr.DataTree:
    """Discover and retrieve in one call, returning an :class:`xarray.DataTree`.

    ``nearest=n`` keeps only the ``n`` closest stations per site per source, which is the usual
    intent when querying a list of locations.

    ``on_error="collect"`` keeps going when a source fails, recording the failures on the tree
    instead of raising. See :meth:`Catalog.fetch` for why the default is strict.
    """
    catalog = discover(
        bbox=bbox,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        sites=sites,
        time=time,
        variables=variables,
        depth=depth,
        providers=providers,
        max_rows=max_rows,
        max_workers=max_workers,
        **options,
    )
    if nearest is not None:
        catalog = catalog.filter(nearest=nearest)
    return catalog.fetch(
        to_cf_units=to_cf_units,
        group_by_site=group_by_site,
        max_workers=max_workers,
        on_error=on_error,
    )


def position(
    lat: float,
    lon: float,
    *,
    radius_km: float = 25.0,
    time: Any = None,
    **kwargs: Any,
) -> xr.DataTree:
    """EDR-shaped sugar: everything within ``radius_km`` of one point."""
    return fetch(lat=lat, lon=lon, radius_km=radius_km, time=time, **kwargs)


def positions(sites: Any, *, radius_km: float = 25.0, time: Any = None, **kwargs: Any):
    """Everything near each of many locations, in one call.

    ``sites`` accepts :class:`Site` objects, ``(lat, lon[, name])`` tuples, dicts, or a
    DataFrame with ``lat``/``lon`` columns — a CSV of moorings works directly. Locations with no
    nearby data simply contribute nothing; use :func:`coverage` to see which those were.
    """
    return fetch(sites=sites, radius_km=radius_km, time=time, **kwargs)


def area(bbox: Sequence[float], *, time: Any = None, **kwargs: Any) -> xr.DataTree:
    """EDR-shaped sugar: everything inside a bounding box."""
    return fetch(bbox=bbox, time=time, **kwargs)
