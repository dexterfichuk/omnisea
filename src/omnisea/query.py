"""The :class:`Query` — omnisea's single, EDR-shaped description of "what do you want?".

The shape deliberately mirrors OGC API - Environmental Data Retrieval: an area (``bbox``) or a
position (``lat``/``lon`` + ``radius_km``), a time interval, a variable list and an optional depth
range. Every provider receives the same ``Query``; a future native EDR adapter is then close to a
straight pass-through.
"""

from __future__ import annotations

import json
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

#: Largest search radius accepted for one site. Half the Earth's circumference is ~20,000 km,
#: so anything approaching this is a slipped decimal point rather than an intention.
MAX_RADIUS_KM = 2000.0

#: Options accepted as keyword arguments, with what each one does. Only the generic knobs live
#: here; every source-specific one — ``resolution``, ``erddap_server``, ``cioos_records`` — is
#: declared by its own provider module through :func:`register_option`, exactly as a
#: third-party provider would declare its own.
#:
#: Validated on construction: a silently-ignored ``resolutio="ONE_MINUTE"`` would hand back
#: minute data the caller believed was hourly, which is exactly the kind of quiet wrongness
#: this library exists to prevent.
KNOWN_OPTIONS: dict[str, str] = {
    "include_unmapped": "keep provider fields that have no CF mapping (default True)",
    "to_cf_units": "convert values to canonical CF units instead of provider units",
    "max_workers": "thread pool size for per-station fetches",
    "max_items": "hard ceiling on features pulled from a paged OGC collection",
}


def register_option(name: str, description: str) -> None:
    """Declare a query option, so a source can accept its own knobs.

    Built-in and third-party sources use the same call, at import time of the module that
    understands the option — the validator cannot know about a knob whose owner is not loaded.
    """
    KNOWN_OPTIONS[name] = description


#: Named keyword parameters of the top-level query functions. Not options, but the most likely
#: thing a misspelled keyword was reaching for — `latitude=` is a far commoner slip than any
#: option typo, and difflib cannot suggest a name it was never shown.
QUERY_KEYWORDS = (
    "bbox", "lat", "lon", "radius_km", "sites", "time", "variables", "depth", "providers",
    "max_rows", "nearest", "group_by_site", "on_error",
)


def _validate_options(options: Mapping[str, Any]) -> dict[str, Any]:
    unknown = [k for k in options if k not in KNOWN_OPTIONS]
    if unknown:
        import difflib

        vocabulary = list(KNOWN_OPTIONS) + list(QUERY_KEYWORDS)
        details = []
        for key in unknown:
            close = difflib.get_close_matches(key, vocabulary, n=1, cutoff=0.55)
            if not close:
                # "latitude" and "lat" score below any sensible cutoff, but a prefix match is
                # unambiguous evidence of what was meant.
                close = [c for c in vocabulary if key.startswith(c) or c.startswith(key)][:1]
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
    # NaT compares False against everything, so the ordering guard below waves it through — and
    # the query then makes zero requests and returns nothing, which reads as "no data here".
    # Reachable straight from a DataFrame cell that failed to parse.
    for label, value in (("start", start_ts), ("end", end_ts)):
        if pd.isna(value):
            raise QueryError(
                f"{label} is not a usable timestamp (got NaT — an empty or unparseable value?)"
            )
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
        # `nan <= 0` is False, so the ordinary guard lets a blank radius_km cell through — and a
        # NaN radius makes a site that matches nothing and contributes a NaN box to the union,
        # silently removing its area from the request. The lat/lon guards above reject NaN by
        # the same comparison logic; this one has to say so explicitly.
        if math.isnan(self.radius_km):
            raise QueryError(
                f"site {self.label!r}: radius_km is NaN (a blank column in a sites CSV?). "
                "Give it a positive radius, or drop the column to take the default."
            )
        if self.radius_km <= 0:
            raise QueryError(
                f"site {self.label!r}: radius_km must be positive; got {self.radius_km}"
            )
        if self.radius_km > MAX_RADIUS_KM:
            # A slipped decimal point matched 1152 stations at only ~49k estimated rows —
            # comfortably under the row ceiling — and became roughly 2,800 requests, which
            # tripped a provider's rate limiter. The row ceiling bounds rows, not requests.
            raise QueryError(
                f"site {self.label!r}: radius_km={self.radius_km:,.0f} is larger than the "
                f"{MAX_RADIUS_KM:,.0f} km ceiling — that is most of the planet, and it becomes "
                "one request per matching station. Use bbox= for a genuinely global query, or "
                "raise omnisea.query.MAX_RADIUS_KM if you mean it."
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
        # A blank cell arrives as float("nan"), which is *truthy* — so `or` chaining would make
        # every unnamed row in a CSV land on the literal label "nan", collapsing them all into
        # one site downstream.
        name = _first_text(lower, ("name", "station", "label"))
        site_id = _first_text(lower, ("id", "site_id")) or None
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


def _first_text(mapping: Mapping[str, Any], candidates: tuple[str, ...]) -> str:
    """The first candidate key holding real text, skipping blanks and NaN."""
    for key in candidates:
        value = mapping.get(key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _reject_duplicate_labels(sites: Sequence[Site]) -> None:
    """Refuse two sites that answer to the same name.

    The label is the join key: it is stamped on every station a site matched, and
    :meth:`Catalog.coverage` and :attr:`Catalog.missing_sites` are keyed by it. Two "Farm A"
    rows in a CSV therefore merge into one — one coverage row for two requested places, and a
    farm that found nothing reported as if it had been covered by the other. Since the whole
    point of a site label is to carry results back to the caller's own records, silently
    merging two of them is worse than asking for distinct names.
    """
    seen: dict[str, int] = {}
    for site in sites:
        seen[site.label] = seen.get(site.label, 0) + 1
    repeated = sorted(label for label, count in seen.items() if count > 1)
    if repeated:
        raise QueryError(
            f"duplicate site label(s): {', '.join(repr(r) for r in repeated)}. Site labels are "
            "the join key results come back on, so two places sharing one name would merge "
            "into a single row. Give each site a distinct name= or id=."
        )


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
        _reject_duplicate_labels(resolved)
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
            # JSON rather than a list: _clean_attrs joins lists with ", " for netCDF, and a
            # label is very often "48.8353,-125.1358" or "Tofino, BC" — re-splitting that on
            # commas invented sites that were never requested and reported them as empty.
            attrs["query_site_names"] = json.dumps([s.label for s in self.sites])
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
    south, north = max(lat - dlat, -90.0), min(lat + dlat, 90.0)
    if abs(dlon) >= 180.0:
        # Near a pole the circle spans every meridian. Clamping dlon to 180 and *then* taking
        # lon +/- dlon kept only half of them, so a station 22 km away across the pole fell
        # outside the box a provider filters on and was never seen. The full range is
        # representable and needs no antimeridian handling, so say so.
        return BBox(-180.0, south, 180.0, north)
    return BBox(max(lon - dlon, -180.0), south, min(lon + dlon, 180.0), north)
