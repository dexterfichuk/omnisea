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
from .catalog import Catalog
from .errors import (
    OmniseaError,
    PayloadTooLargeError,
    ProviderError,
    QueryError,
    UnknownProviderError,
    UpstreamError,
)
from .http import DEFAULT_MAX_WORKERS, map_threads, set_max_concurrency
from .providers import BUILTIN_PROVIDERS
from .providers.base import (
    DataSource,
    DiscoverySource,
    Provider,
    RetrievalSource,
    StationMatch,
    StationSeries,
)
from .query import Query, Site, as_sites, register_option
from .registry import register_provider, register_source
from .tree import build_tree, coverage, stations, summary, to_dataframe

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
    "Catalog",
    # introspection
    "providers",
    "sources",
    "variables",
    # tree helpers
    "summary",
    "stations",
    "to_dataframe",
    "coverage",
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
    "set_max_concurrency",
    # errors
    "OmniseaError",
    "QueryError",
    "ProviderError",
    "UpstreamError",
    "UnknownProviderError",
    "PayloadTooLargeError",
]

for _provider in BUILTIN_PROVIDERS:
    registry.register_provider(_provider)


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


def variables() -> pd.DataFrame:
    """CF standard names omnisea can serve, and which sources serve each one.

    This lists the *canonical* names only. Every source also returns its unmapped fields under
    their original provider names, so this is a floor on what you get, not a ceiling.
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
    return frame.sort_values(["variable", "source"]).reset_index(drop=True)


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

    def _discover(source: DataSource) -> list[StationMatch]:
        try:
            return source.discover(query)
        except Exception as exc:  # noqa: BLE001 - one dead API must not sink the rest
            errors[source.name] = f"{type(exc).__name__}: {exc}"
            log.warning("discovery failed for %s: %s", source.name, exc)
            return []

    matches: list[StationMatch] = []
    for found in map_threads(_discover, selected, max_workers=max_workers, label="discovery"):
        matches.extend(found)

    return Catalog(query, matches, errors)


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
    **options: Any,
) -> xr.DataTree:
    """Discover and retrieve in one call, returning an :class:`xarray.DataTree`.

    ``nearest=n`` keeps only the ``n`` closest stations per site per source, which is the usual
    intent when querying a list of locations.
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
