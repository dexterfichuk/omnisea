"""The :class:`Query` — omnisea's single, EDR-shaped description of "what do you want?".

The shape deliberately mirrors OGC API - Environmental Data Retrieval: an area (``bbox``) or a
position (``lat``/``lon`` + ``radius_km``), a time interval, a variable list and an optional depth
range. Every provider receives the same ``Query``; a future native EDR adapter is then close to a
straight pass-through.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, NamedTuple

import pandas as pd

from .errors import QueryError

__all__ = ["Query", "Site", "BBox", "to_utc", "as_sites", "register_option",
           "KNOWN_OPTIONS"]

class BBox(NamedTuple):
    """A geographic bounding box, in the OGC order ``(west, south, east, north)``.

    A ``NamedTuple`` rather than a bare 4-tuple so that call sites can say ``bbox.south``
    instead of ``bbox[1]``, and so a reader can tell at a glance which convention is in play —
    lon-first (OGC, GeoJSON, EDR) rather than the lat-first order people often reach for.
    It still unpacks, indexes and compares as an ordinary tuple, so plain tuples remain
    acceptable input everywhere.
    """

    west: float
    south: float
    east: float
    north: float

    @property
    def centre(self) -> tuple[float, float]:
        """``(lat, lon)`` of the box centre."""
        return ((self.south + self.north) / 2, (self.west + self.east) / 2)

EARTH_RADIUS_KM = 6371.0088

#: Source-specific knobs accepted as keyword arguments, with what each one does.
#: Validated on construction: a silently-ignored ``resolutio="ONE_MINUTE"`` would hand back
#: minute data the caller believed was hourly, which is exactly the kind of quiet wrongness
#: this library exists to prevent.
KNOWN_OPTIONS: dict[str, str] = {
    "include_unmapped": "keep provider fields that have no CF mapping (default True)",
    "to_cf_units": "convert values to canonical CF units instead of provider units",
    "max_workers": "thread pool size for per-station fetches",
    "max_items": "hard ceiling on features pulled from a paged OGC collection",
    "resolution": "dfo_tides: ONE_MINUTE | THREE_MINUTES | FIVE_MINUTES | "
    "FIFTEEN_MINUTES | SIXTY_MINUTES",
    "series": "dfo_tides: which IWLS series to pull, e.g. ('wlo', 'wlp', 'wlp-hilo')",
    "erddap_server": "erddap: ERDDAP server root URL (default CIOOS Pacific)",
    "erddap_datasets": "erddap: dataset id(s) to use instead of searching the server",
    "erddap_search": "erddap: free-text searchFor passed to the ERDDAP search index",
    "erddap_max_datasets": "erddap: ceiling on datasets discovery will describe (default 25)",
    "cioos_records": "cioos_metadata: path, directory, URL or owner/repo holding metadata records",
    "cioos_token": "cioos_metadata: token for an authenticated records endpoint",
}


def register_option(name: str, description: str) -> None:
    """Declare a query option, so third-party sources can accept their own knobs."""
    KNOWN_OPTIONS[name] = description


def _validate_options(options: Mapping[str, Any]) -> dict[str, Any]:
    unknown = [k for k in options if k not in KNOWN_OPTIONS]
    if unknown:
        import difflib

        details = []
        for key in unknown:
            close = difflib.get_close_matches(key, KNOWN_OPTIONS, n=1)
            details.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise QueryError(
            "unknown option(s): "
            + ", ".join(details)
            + "\n  known options: "
            + ", ".join(sorted(KNOWN_OPTIONS))
        )
    return dict(options)


def to_utc(value: Any, *, label: str = "time") -> pd.Timestamp:
    """Coerce anything time-like to a tz-aware UTC :class:`pandas.Timestamp`.

    Naive input is *assumed* UTC rather than local — marine APIs speak UTC, and silently applying
    the caller's machine timezone is the kind of bug that only shows up in someone else's timezone.
    """
    if value is None:
        raise QueryError(f"{label} must not be None")
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:  # noqa: BLE001 - re-raised as a library error
        raise QueryError(f"could not interpret {label}={value!r} as a timestamp: {exc}") from exc
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _normalize_time(time: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Accept ``(start, end)``, a slice, or a single day and return a UTC half-open interval."""
    if isinstance(time, slice):
        start, end = time.start, time.stop
    elif isinstance(time, (str, datetime, date, pd.Timestamp)):
        # A bare day means that whole day.
        start = to_utc(time, label="time")
        end = start + pd.Timedelta(days=1)
        return start, end
    elif isinstance(time, Sequence) and len(time) == 2:
        start, end = time
    else:
        raise QueryError(
            "time must be (start, end), a slice, or a single date; " f"got {time!r}"
        )
    start_ts = to_utc(start, label="start")
    end_ts = to_utc(end, label="end")
    if end_ts <= start_ts:
        raise QueryError(f"end ({end_ts}) must be after start ({start_ts})")
    return start_ts, end_ts


def _normalize_bbox(bbox: Sequence[float]) -> BBox:
    if len(bbox) != 4:
        raise QueryError(f"bbox must be (west, south, east, north); got {bbox!r}")
    west, south, east, north = (float(v) for v in bbox)
    if south > north:
        raise QueryError(f"bbox south ({south}) is north of north ({north})")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise QueryError(f"bbox latitudes must lie in [-90, 90]; got {south}, {north}")
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise QueryError(f"bbox longitudes must lie in [-180, 180]; got {west}, {east}")
    if west > east:
        raise QueryError(
            f"bbox west ({west}) is east of east ({east}); "
            "antimeridian-crossing boxes are not supported yet"
        )
    return BBox(west, south, east, north)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class Site:
    """One named point of interest, with its own search radius.

    A "site" is the unit of a multi-location request: you hand omnisea a list of moorings, farms
    or sampling stations and each one carries the label you already use for it, so results come
    back joinable to your own table without a second lookup.
    """

    lat: float
    lon: float
    name: str = ""
    radius_km: float = 25.0
    id: str | None = None

    def __post_init__(self) -> None:
        if not -90 <= self.lat <= 90:
            raise QueryError(f"site {self.label!r}: lat must lie in [-90, 90]; got {self.lat}")
        if not -180 <= self.lon <= 180:
            raise QueryError(f"site {self.label!r}: lon must lie in [-180, 180]; got {self.lon}")
        if self.radius_km <= 0:
            raise QueryError(
                f"site {self.label!r}: radius_km must be positive; got {self.radius_km}"
            )

    @property
    def label(self) -> str:
        """Whatever the caller can recognise this site by."""
        return self.name or self.id or f"{self.lat:.4f},{self.lon:.4f}"

    @property
    def bbox(self) -> BBox:
        return _bbox_around(self.lat, self.lon, self.radius_km)

    def distance_km(self, lat: float, lon: float) -> float:
        return haversine_km(self.lat, self.lon, lat, lon)


def as_sites(obj: Any, *, default_radius_km: float = 25.0) -> tuple[Site, ...]:
    """Coerce many spellings of "a list of places" into :class:`Site` objects.

    Accepts :class:`Site` instances, ``(lat, lon)`` / ``(lat, lon, name)`` tuples, mappings with
    ``lat``/``lon`` keys (plus optional ``name``/``id``/``radius_km``), or a
    :class:`pandas.DataFrame` with ``lat``/``lon`` columns — because the realistic input is a CSV
    of sites someone already keeps, not hand-written Python.
    """
    if obj is None:
        return ()
    if isinstance(obj, Site):
        return (obj,)

    # A DataFrame of sites is the common real-world case; take it as rows of mappings.
    if isinstance(obj, pd.DataFrame):
        rows = obj.to_dict("records")
        return tuple(_as_site(r, default_radius_km) for r in rows)

    if isinstance(obj, Mapping):
        return (_as_site(obj, default_radius_km),)

    # A bare (lat, lon) pair is a single site, not two sites.
    if (
        isinstance(obj, Sequence)
        and not isinstance(obj, (str, bytes))
        and len(obj) in (2, 3)
        and all(isinstance(v, (int, float)) for v in obj[:2])
        and not isinstance(obj[0], bool)
    ):
        return (_as_site(obj, default_radius_km),)

    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return tuple(_as_site(item, default_radius_km) for item in obj)

    raise QueryError(f"could not interpret {obj!r} as a site or list of sites")


def _as_site(item: Any, default_radius_km: float) -> Site:
    if isinstance(item, Site):
        return item
    if isinstance(item, Mapping):
        lower = {str(k).lower(): v for k, v in item.items()}
        try:
            lat = float(lower[_first_key(lower, ("lat", "latitude", "y"))])
            lon = float(lower[_first_key(lower, ("lon", "long", "longitude", "x"))])
        except (KeyError, TypeError, ValueError) as exc:
            raise QueryError(
                f"site mapping needs numeric lat/lon keys; got {dict(item)!r}"
            ) from exc
        name = lower.get("name") or lower.get("station") or lower.get("label") or ""
        site_id = lower.get("id") or lower.get("site_id") or None
        radius = lower.get("radius_km", default_radius_km)
        return Site(
            lat=lat,
            lon=lon,
            name="" if name is None else str(name),
            radius_km=float(default_radius_km if radius is None else radius),
            id=None if site_id is None else str(site_id),
        )
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
        if len(item) == 2:
            return Site(lat=float(item[0]), lon=float(item[1]), radius_km=default_radius_km)
        if len(item) == 3:
            return Site(
                lat=float(item[0]),
                lon=float(item[1]),
                name=str(item[2]),
                radius_km=default_radius_km,
            )
        raise QueryError(f"site tuple must be (lat, lon) or (lat, lon, name); got {item!r}")
    raise QueryError(f"could not interpret {item!r} as a site")


def _first_key(mapping: Mapping[str, Any], candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in mapping:
            return c
    raise KeyError(candidates[0])


def _union_bbox(boxes: Sequence[BBox]) -> BBox:
    """Smallest box covering them all — the pre-filter sent upstream for a multi-site query."""
    return BBox(
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


@dataclass(frozen=True)
class Query:
    """An immutable, UTC-normalized spatial-temporal request.

    Build one with :meth:`from_area`, :meth:`from_position` or :meth:`from_sites` rather than the
    raw constructor — they do the validation and the position-to-bbox expansion.

    Space is expressed one of two ways. A pure *area* query carries only ``bbox``. A *site* query
    carries one or more :class:`Site` points, each with its own radius; ``bbox`` is then the union
    of their boxes and serves only as the coarse filter sent upstream, with the per-site radius
    deciding what actually matches.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    bbox: BBox | None = None
    sites: tuple[Site, ...] = ()
    #: CF names, omnisea variable names or raw provider field names used to choose *which
    #: sources and stations to fetch*. It is not a projection: whatever is fetched comes back
    #: with every field the platform published, because the response already contains them.
    variables: frozenset[str] | None = None
    depth: tuple[float, float] | None = None
    providers: tuple[str, ...] | None = None
    max_rows: int = 2_000_000
    options: Mapping[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- constructors

    @classmethod
    def from_area(
        cls,
        bbox: Sequence[float],
        time: Any,
        *,
        variables: Iterable[str] | None = None,
        depth: Sequence[float] | None = None,
        providers: Iterable[str] | None = None,
        max_rows: int = 2_000_000,
        **options: Any,
    ) -> Query:
        start, end = _normalize_time(time)
        return cls(
            start=start,
            end=end,
            bbox=_normalize_bbox(bbox),
            variables=frozenset(variables) if variables else None,
            depth=(float(depth[0]), float(depth[1])) if depth else None,
            providers=tuple(providers) if providers else None,
            max_rows=max_rows,
            options=_validate_options(options),
        )

    @classmethod
    def from_position(
        cls,
        lat: float,
        lon: float,
        time: Any,
        *,
        radius_km: float = 25.0,
        name: str = "",
        variables: Iterable[str] | None = None,
        depth: Sequence[float] | None = None,
        providers: Iterable[str] | None = None,
        max_rows: int = 2_000_000,
        **options: Any,
    ) -> Query:
        site = Site(lat=float(lat), lon=float(lon), name=name, radius_km=float(radius_km))
        return cls.from_sites(
            [site],
            time,
            variables=variables,
            depth=depth,
            providers=providers,
            max_rows=max_rows,
            **options,
        )

    @classmethod
    def from_sites(
        cls,
        sites: Any,
        time: Any,
        *,
        radius_km: float = 25.0,
        variables: Iterable[str] | None = None,
        depth: Sequence[float] | None = None,
        providers: Iterable[str] | None = None,
        max_rows: int = 2_000_000,
        **options: Any,
    ) -> Query:
        """Build a query over many places at once.

        ``sites`` may be :class:`Site` objects, ``(lat, lon[, name])`` tuples, dicts, or a
        DataFrame with ``lat``/``lon`` columns. ``radius_km`` is the default for sites that do not
        carry their own.
        """
        resolved = as_sites(sites, default_radius_km=radius_km)
        if not resolved:
            raise QueryError("no sites given; pass at least one (lat, lon)")
        start, end = _normalize_time(time)
        return cls(
            start=start,
            end=end,
            bbox=_union_bbox([s.bbox for s in resolved]),
            sites=resolved,
            variables=frozenset(variables) if variables else None,
            depth=(float(depth[0]), float(depth[1])) if depth else None,
            providers=tuple(providers) if providers else None,
            max_rows=max_rows,
            options=_validate_options(options),
        )

    # ---------------------------------------------------------------- geometry

    @property
    def is_multi_site(self) -> bool:
        return len(self.sites) > 1

    @property
    def lat(self) -> float | None:
        """Latitude of the single site, or ``None`` for area / multi-site queries."""
        return self.sites[0].lat if len(self.sites) == 1 else None

    @property
    def lon(self) -> float | None:
        return self.sites[0].lon if len(self.sites) == 1 else None

    @property
    def radius_km(self) -> float | None:
        return self.sites[0].radius_km if len(self.sites) == 1 else None

    def nearest_site(self, lat: float, lon: float) -> tuple[Site, float] | None:
        """The closest site to a point, with its distance, or ``None`` for an area query."""
        if not self.sites:
            return None
        best = min(self.sites, key=lambda s: s.distance_km(lat, lon))
        return best, best.distance_km(lat, lon)

    def distance_km(self, lat: float, lon: float) -> float | None:
        """Distance to the nearest site, or ``None`` for a pure bbox query."""
        nearest = self.nearest_site(lat, lon)
        return None if nearest is None else nearest[1]

    def contains(self, lat: float, lon: float) -> bool:
        """Is this point inside the requested area?

        For a site query the union bbox is only the coarse pre-filter — a point must fall within
        some individual site's radius, so a station sitting in the empty corner between two
        far-apart sites is correctly excluded.
        """
        if self.sites:
            return any(s.distance_km(lat, lon) <= s.radius_km for s in self.sites)
        if self.bbox is not None:
            west, south, east, north = self.bbox
            return south <= lat <= north and west <= lon <= east
        return True

    # ---------------------------------------------------------------- time

    @property
    def days(self) -> float:
        return (self.end - self.start) / pd.Timedelta(days=1)

    def overlaps(self, first: Any, last: Any) -> bool:
        """Does a station's period of record intersect the query window?

        ``None`` bounds mean "open-ended", which is how the ECCC station catalogue marks a station
        that is still reporting.
        """
        try:
            f = to_utc(first) if first is not None else None
            latest = to_utc(last) if last is not None else None
        except QueryError:
            return False
        if f is not None and f >= self.end:
            return False
        if latest is not None and latest <= self.start:
            return False
        return True

    # ---------------------------------------------------------------- misc

    def wants(self, *names: str) -> bool:
        """True when the caller asked for one of ``names`` (or did not restrict variables)."""
        if self.variables is None:
            return True
        return any(n in self.variables for n in names)

    def option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def replace(self, **changes: Any) -> Query:
        return replace(self, **changes)

    @property
    def interval_iso(self) -> str:
        """The window as an OGC ``datetime`` interval string."""
        return (
            f"{self.start.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
            f"{self.end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )

    def to_attrs(self) -> dict[str, Any]:
        """Flat, netCDF-safe description of this query, recorded on the tree root."""
        attrs: dict[str, Any] = {
            "query_start": self.start.isoformat(),
            "query_end": self.end.isoformat(),
        }
        if self.bbox is not None:
            attrs["query_bbox"] = list(self.bbox)
        if self.sites:
            attrs["query_site_names"] = [s.label for s in self.sites]
            attrs["query_site_lats"] = [s.lat for s in self.sites]
            attrs["query_site_lons"] = [s.lon for s in self.sites]
            attrs["query_site_radius_km"] = [s.radius_km for s in self.sites]
        if self.variables:
            attrs["query_variables"] = sorted(self.variables)
        if self.depth is not None:
            attrs["query_depth"] = list(self.depth)
        if self.providers:
            attrs["query_providers"] = list(self.providers)
        return attrs

    def __repr__(self) -> str:  # pragma: no cover - display only
        if len(self.sites) == 1:
            s = self.sites[0]
            where = f"position({s.lat}, {s.lon}, r={s.radius_km}km)"
        elif self.sites:
            where = f"{len(self.sites)} sites"
        else:
            where = f"bbox{self.bbox}"
        var = ",".join(sorted(self.variables)) if self.variables else "all"
        return (
            f"Query({where}, {self.start:%Y-%m-%d %H:%M}Z -> {self.end:%Y-%m-%d %H:%M}Z, "
            f"variables={var})"
        )


def _bbox_around(lat: float, lon: float, radius_km: float) -> BBox:
    """Smallest lat/lon box enclosing the radius circle, clamped to valid ranges."""
    dlat = math.degrees(radius_km / EARTH_RADIUS_KM)
    # Longitude degrees shrink with latitude; guard the poles where cos -> 0.
    coslat = math.cos(math.radians(lat))
    dlon = 180.0 if abs(coslat) < 1e-9 else math.degrees(radius_km / (EARTH_RADIUS_KM * coslat))
    dlon = min(abs(dlon), 180.0)
    return BBox(
        max(lon - dlon, -180.0),
        max(lat - dlat, -90.0),
        min(lon + dlon, 180.0),
        min(lat + dlat, 90.0),
    )
