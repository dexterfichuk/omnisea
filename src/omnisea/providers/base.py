"""The provider/source class hierarchy — omnisea's extension point.

Two levels, because real marine data has two levels:

* A :class:`Provider` is an **organization or service**: DFO, ECCC, CIOOS. It owns the base URL,
  the licence, the attribution and any auth, and it is what you credit in a paper.
* A :class:`DataSource` is **one queryable dataset** belonging to a provider: IWLS tides,
  ``climate-hourly``, ``swob-realtime``. It owns the field table, the node path and the actual
  discover/fetch logic.

ECCC alone publishes four datasets that share paging, bbox handling and GeoJSON decoding but
differ in their fields and time semantics — that shared middle is what the hierarchy captures.
Adding a dataset to an existing provider is a subclass; adding a new organization is one new
module.

Users select :class:`DataSource` names (``"dfo_tides"``, ``"eccc_climate"``), and naming a
provider (``"eccc"``) selects all of its sources.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import xarray as xr

from .. import cf
from ..errors import ProviderError
from ..http import get_json
from ..query import Query

__all__ = [
    "drop_orphan_qc",
    "trim_to_window",
    "Provider",
    "DataSource",
    "RetrievalSource",
    "DiscoverySource",
    "StationMatch",
    "StationSeries",
    "frame_from_records",
]

log = logging.getLogger("omnisea.providers")


# --------------------------------------------------------------------------- data carriers


@dataclass
class StationMatch:
    """One candidate station found during discovery — a row in the :class:`~omnisea.Catalog`.

    Discovery is deliberately cheap: it answers "what is there, and roughly how much of it?" so
    the caller can look before committing to a download.
    """

    source: str
    station_id: str
    name: str
    lat: float
    lon: float
    variables: tuple[str, ...] = ()
    n_rows_est: int = 0
    distance_km: float | None = None
    site: str | None = None  # label of the nearest requested site, for multi-site queries
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    provider: str = ""  # organization that owns the source, e.g. "eccc"
    #: Source-private payload handed from :meth:`DataSource.discover` to
    #: :meth:`DataSource.fetch` — internal ids, series codes, anything the fetch step needs
    #: that the catalogue itself should not display. Read required keys with :meth:`require`.
    extra: dict[str, Any] = field(default_factory=dict)

    def require(self, key: str) -> Any:
        """Read a value ``discover()`` promised to ``fetch()``, failing loudly if absent.

        ``extra`` is deliberately untyped — each source needs different things in it — which
        makes it the one place a typo can quietly cost you a station. Reading through here turns
        that into a named error instead of a silently missing series, which for scientific data
        is the difference between a bug you find and one you publish.
        """
        try:
            return self.extra[key]
        except KeyError:
            raise ProviderError(
                f"discovery did not record {key!r} for station {self.station_id!r}, "
                f"so it cannot be fetched. This is a bug in the {self.source} adapter, "
                f"not in your query. Available keys: {sorted(self.extra) or '(none)'}",
                provider=self.source,
            ) from None

    def attach_site(self, query: Query) -> StationMatch:
        """Record which requested site this station answers for, and how far away it is."""
        nearest = query.nearest_site(self.lat, self.lon)
        if nearest is not None:
            site, distance = nearest
            self.site = site.label
            self.distance_km = distance
        return self

    def as_row(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "provider": self.provider,
            "site": self.site,
            "station_id": self.station_id,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "distance_km": self.distance_km,
            "variables": ", ".join(self.variables),
            "n_rows_est": self.n_rows_est,
            "first": self.first,
            "last": self.last,
        }


@dataclass
class StationSeries:
    """A fetched point time series, ready to become one node of the tree.

    ``frame`` is indexed by a tz-aware UTC ``DatetimeIndex`` named ``time``; its columns are
    already CF-named where a CF name exists, with QC flags carried alongside as ``<var>_qc``.
    """

    match: StationMatch
    frame: pd.DataFrame
    node_path: str
    attrs: dict[str, Any] = field(default_factory=dict)
    var_attrs: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.frame is None or self.frame.empty


# --------------------------------------------------------------------------- provider


class Provider(ABC):
    """An organization or service that publishes one or more datasets.

    Subclasses declare the identity and licensing that every node from every one of their
    sources will carry, and instantiate their sources in :meth:`build_sources`.
    """

    #: Short registry key for the organization, e.g. ``"eccc"``.
    name: str = ""
    #: Human-readable organization name, used in attribution.
    title: str = ""
    #: Root URL all of this provider's sources hang off.
    base_url: str = ""
    #: Licence string recorded on every node.
    license: str = ""
    #: Where the licence and terms live.
    terms_url: str = ""

    def __init__(self) -> None:
        self._sources: list[DataSource] | None = None

    @abstractmethod
    def build_sources(self) -> Sequence[DataSource]:
        """Instantiate this provider's datasets. Called once, lazily."""

    @property
    def sources(self) -> list[DataSource]:
        if self._sources is None:
            self._sources = list(self.build_sources())
        return self._sources

    # ------------------------------------------------------------------ http

    def get_json(
        self, path: str, params: Mapping[str, Any] | None = None, *, source: str | None = None
    ) -> Any:
        """GET JSON relative to :attr:`base_url` (or an absolute URL), tagged with this provider."""
        url = path if path.startswith("http") else f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        return get_json(url, dict(params or {}), provider=source or self.name)

    def attribution(self) -> dict[str, Any]:
        """Identity/licence attributes stamped onto every node this provider produces."""
        return {
            "institution": self.title or self.name,
            "license": self.license,
            "references": self.terms_url,
            "provider": self.name,
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<Provider {self.name!r} sources={[s.name for s in self.sources]}>"


# --------------------------------------------------------------------------- source


class DataSource(ABC):
    """One queryable dataset. This is what users select and what the registry keys on."""

    #: Registry key, e.g. ``"eccc_climate"``.
    name: str = ""
    #: Human-readable dataset name.
    title: str = ""
    #: Where this source's nodes live in the tree, e.g. ``"in_situ/weather"``.
    node_path: str = ""
    #: CF discrete-sampling-geometry type: ``timeSeries``, ``trajectory``, ``profile``, ``grid``.
    feature_type: str = "timeSeries"
    #: True for sources that describe data rather than serve it (Catalog rows only, no arrays).
    discovery_only: bool = False
    #: This dataset's raw-field to CF mapping. Defined on the class, next to the code that uses
    #: it, so adding a source is one new file rather than an edit in two places.
    fields: dict[str, cf.FieldSpec] = {}
    #: pandas **Period** alias ("D", "M", "Y") for collections whose rows summarize a period
    #: and are labelled by the period's first instant.
    #:
    #: Without it, asking for "15 July noon to 17 July" silently returns fewer days than
    #: "15 July to 17 July": the 15 July summary is stamped 00:00Z, so the trim discards it even
    #: though the day it describes overlaps the request. A period belongs to a window when its
    #: *interval* overlaps, not when its label instant happens to fall inside.
    period: str | None = None
    #: True when the source reads its field descriptions from the data at runtime rather than
    #: declaring them — an ERDDAP dataset publishes its own standard names and units, and
    #: hardcoding them would be both wrong and unmaintainable. An empty ``fields`` is then a
    #: design choice, not an omission.
    fields_from_metadata: bool = False
    #: Groups of raw fields that mean the same thing and never appear in one record — a
    #: platform spelling a measurement two ways across hardware generations, say. They may
    #: share an output variable; without this a reviewer cannot tell that from a mistake.
    equivalent_fields: tuple[frozenset[str], ...] = ()
    #: How far back this dataset holds data, for the "realtime" collections that keep only a
    #: rolling window. ``None`` means a full historical archive.
    #:
    #: This exists so that asking for 2024 river levels does not come back as a silent empty
    #: tree, which reads as "there is no gauge here" when the truth is "this collection only
    #: keeps 30 days, and the historical one is a different source".
    retention: pd.Timedelta | None = None

    def __init__(self, provider: Provider):
        self.provider = provider

    # ------------------------------------------------------------------ description

    @property
    def variables(self) -> frozenset[str]:
        """CF standard names this source can serve.

        Unmapped fields are still returned by :meth:`fetch` under their provider names; this
        lists only what has a canonical name to advertise.
        """
        return frozenset(
            spec.standard_name or spec.var for spec in self.fields.values()
        )

    # ------------------------------------------------------------------ contract

    @abstractmethod
    def discover(self, query: Query) -> list[StationMatch]:
        """Stations this source can offer for ``query``. Cheap; no bulk data transfer."""

    # ------------------------------------------------------------------ helpers

    def new_match(self, **kwargs: Any) -> StationMatch:
        """A :class:`StationMatch` pre-tagged with this source and its provider."""
        kwargs.setdefault("source", self.name)
        kwargs.setdefault("provider", self.provider.name)
        return StationMatch(**kwargs)

    def base_attrs(self, **extra: Any) -> dict[str, Any]:
        """Node attributes every series from this source carries."""
        attrs: dict[str, Any] = {
            "Conventions": "CF-1.10",
            "featureType": self.feature_type,
            "source_name": self.name,
            **self.provider.attribution(),
        }
        attrs.update({k: v for k, v in extra.items() if v is not None})
        return attrs

    def retention_cutoff(self, now: pd.Timestamp | None = None) -> pd.Timestamp | None:
        """The earliest instant this source can still serve, or ``None`` if it keeps everything."""
        if self.retention is None:
            return None
        return (now or pd.Timestamp.now(tz="UTC")) - self.retention

    def retention_gap(self, query: Query, now: pd.Timestamp | None = None) -> str | None:
        """A plain explanation if the query reaches past what this source keeps.

        Returns ``None`` when the window is fully covered. Otherwise the message says what the
        source holds and what to use instead, because "no results" is the least useful possible
        answer to "give me last year's river levels".
        """
        cutoff = self.retention_cutoff(now)
        if cutoff is None:
            return None
        days = int(self.retention / pd.Timedelta(days=1))
        if query.end <= cutoff:
            return (
                f"holds only the last ~{days} days (back to {cutoff:%Y-%m-%d}); the requested "
                f"window ends {query.end:%Y-%m-%d} and is entirely outside it"
            )
        if query.start < cutoff:
            return (
                f"holds only the last ~{days} days (back to {cutoff:%Y-%m-%d}); the earlier "
                f"part of the requested window is not available from this source"
            )
        return None

    def covers(self, query: Query, now: pd.Timestamp | None = None) -> bool:
        """False when the window lies entirely outside this source's retention."""
        cutoff = self.retention_cutoff(now)
        return cutoff is None or query.end > cutoff

    def recognizes(self, name: str) -> bool:
        """Is ``name`` one of this source's *curated* names (CF, omnisea, or raw field)?"""
        for raw, spec in self.fields.items():
            if name in (raw, spec.var) or (spec.standard_name and name == spec.standard_name):
                return True
        return False

    def wants_anything(self, query: Query) -> bool:
        """Could this source serve any of the requested variables?

        Used to skip sources that plainly cannot help, so a tide query does not hit four weather
        collections. The subtlety is that a curated CF table is a *floor*, not an inventory:
        SWOB publishes about 74 fields and omnisea names 12 of them, and the rest come back as
        passthrough. So an unrecognised name is not evidence the source lacks it — omnisea
        simply cannot know without fetching.

        The rule therefore is: match on the curated table when we can, and when a requested name
        is one omnisea does not recognise *anywhere*, stay in rather than opt out. Being asked
        for ``batry_volt`` and answering "no such variable" would be wrong; the field exists,
        omnisea just has no CF name for it.
        """
        if query.variables is None:
            return True
        wanted = cf.resolve_names(query.variables) or frozenset()
        if any(self.recognizes(name) for name in wanted):
            return True

        from ..registry import known_variable_names

        known = known_variable_names()
        return any(name not in known for name in wanted)

    def period_window(self, query: Query) -> tuple[pd.Timestamp, pd.Timestamp]:
        """The query window grown out to whole :attr:`period` aggregation periods.

        Needed at both ends of the round trip. Upstream matches an aggregate against the period
        it covers, so a window landing inside one period matches nothing:
        ``hydrometric-annual-statistics`` returns 0 rows for ``2020-06-01/2020-09-30`` and 2 for
        ``2020-01-01/2020-12-31``. And a row that did come back would then be trimmed away for
        being stamped before the window opened, since a period is labelled by its first instant.
        Growing both ends keeps every period that overlaps what was asked for.

        Idempotent, so a source may widen its request and still let the shared trim run.
        """
        start = query.start.tz_convert("UTC").tz_localize(None).to_period(self.period)
        end = query.end.tz_convert("UTC").tz_localize(None).to_period(self.period)
        return start.start_time.tz_localize("UTC"), end.end_time.tz_localize("UTC")

    def trim_window(self, query: Query) -> tuple[pd.Timestamp, pd.Timestamp]:
        """The window the fetched frame is trimmed to."""
        if not self.period:
            return query.start, query.end
        return self.period_window(query)

    def include_unmapped(self, query: Query) -> bool:
        """Whether to carry fields that have no CF mapping (default: yes, keep everything)."""
        return bool(query.option("include_unmapped", True))

    def to_cf_units(self, query: Query) -> bool:
        """Whether to emit canonical CF units instead of the provider's own (default: no).

        The conversion happens where the values are read, so the ``units`` attribute and the
        numbers beside it can never disagree.
        """
        return bool(query.option("to_cf_units", False))

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<{type(self).__name__} name={self.name!r} node={self.node_path!r}>"


# --------------------------------------------------------------------------- utilities


def frame_from_records(
    records: list[Mapping[str, Any]], *, time_key: str = "time"
) -> pd.DataFrame:
    """Build a time-indexed frame from row dicts, sorted and de-duplicated.

    Duplicate timestamps are expected — chunked requests share their boundary instants — so the
    last value for a timestamp wins and the index comes back strictly increasing.
    """
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    if time_key not in frame.columns:
        return pd.DataFrame()
    frame[time_key] = pd.to_datetime(frame[time_key], utc=True, format="mixed", errors="coerce")
    frame = frame.dropna(subset=[time_key])
    if frame.empty:
        return pd.DataFrame()
    frame = frame.set_index(time_key).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.index.name = "time"
    # Columns that are entirely empty carry no information and clutter every node.
    frame = frame.dropna(axis=1, how="all")
    return frame


class RetrievalSource(DataSource):
    """A source that actually returns data arrays."""

    discovery_only = False

    @abstractmethod
    def fetch(
        self, query: Query, matches: list[StationMatch]
    ) -> list[StationSeries | xr.Dataset]:
        """Pull the data for the confirmed subset of ``matches``.

        Returning either :class:`StationSeries` (the point path, assembled by ``tree.py``) or a
        ready-made :class:`xarray.Dataset` (the gridded path) is the seam that lets a future
        Copernicus or griddap source drop in without the point-assembly code knowing about it.
        """


class DiscoverySource(DataSource):
    """A source that describes data without serving it.

    Metadata catalogues — CIOOS records, STAC, and eventually any ISO 19115 endpoint — tell you
    *what exists and where*, and hand back a URL rather than an array. They contribute
    :class:`~omnisea.Catalog` rows and nothing to the tree, so they get their own base class
    instead of an inherited ``fetch`` that could only ever return an empty list.
    """

    discovery_only = True

    def fetch(
        self, query: Query, matches: list[StationMatch]
    ) -> list[StationSeries | xr.Dataset]:
        """Discovery sources contribute catalogue rows only."""
        return []


def drop_orphan_qc(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove ``<var>_qc`` columns whose measurement column is gone.

    A variable that was entirely empty gets dropped, but its flag column can survive (ECCC marks
    missing values with an ``M`` flag on every row, so the flags are *not* empty). A lone
    ``precipitation_amount_qc`` with no ``precipitation_amount`` beside it is just confusing.
    """
    if frame is None or frame.empty:
        return frame
    orphans = [
        col
        for col in frame.columns
        if str(col).endswith("_qc") and str(col)[: -len("_qc")] not in frame.columns
    ]
    return frame.drop(columns=orphans) if orphans else frame


def trim_to_window(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Clip a time-indexed frame to the requested window, inclusive of both endpoints.

    Upstream filters do not always mean what the caller meant — ECCC filters ``climate-hourly``
    on *local* dates while omnisea labels rows in UTC — so the window is enforced here. Asking
    for a week and receiving a week shifted by the station's UTC offset is exactly the kind of
    quiet wrongness this library exists to prevent.
    """
    if frame is None or frame.empty:
        return frame
    return frame[(frame.index >= start) & (frame.index <= end)]
