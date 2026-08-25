"""Shared machinery for OGC API - Features (pygeoapi) collections.

Everything ECCC publishes — hourly climate, daily climate, SWOB, hydrometric — is served by the
same pygeoapi instance with the same paging, the same GeoJSON envelope and the same
``bbox``/``datetime`` filters. That commonality lives here; each subclass declares only what is
genuinely different about its dataset: the collection id, where the time comes from, how a
station is identified, and its field table.

One rule is enforced for every collection: **latitude and longitude come from
``geometry.coordinates``, never from ``LATITUDE``/``LONGITUDE`` properties**, because
``climate-stations`` publishes those as integer micro-degrees (``483300000`` for 48.33°N) and
reading them naively puts every station in the same wrong place.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from .. import cf
from ..http import DEFAULT_MAX_WORKERS, map_threads, paginate_ogc_items
from ..query import Query
from .base import (
    Provider,
    RetrievalSource,
    StationMatch,
    StationSeries,
    drop_orphan_qc,
    frame_from_records,
    trim_to_window,
)

log = logging.getLogger("omnisea.ogc")

__all__ = ["OgcFeaturesProvider", "OgcFeaturesSource", "point_from_feature"]


def point_from_feature(feature: Mapping[str, Any]) -> tuple[float, float] | None:
    """``(lat, lon)`` from a GeoJSON feature's geometry, or ``None`` if it has no usable point.

    Always the geometry — see the module docstring for why the properties cannot be trusted.
    """
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates")
    if not isinstance(coords, Sequence) or len(coords) < 2:
        return None
    try:
        lon, lat = float(coords[0]), float(coords[1])
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


class OgcFeaturesProvider(Provider):
    """A provider whose datasets are OGC API - Features collections."""

    def collection_url(self, collection: str) -> str:
        return f"{self.base_url.rstrip('/')}/collections/{collection}/items"


class OgcFeaturesSource(RetrievalSource):
    """One OGC API - Features collection of station observations."""

    #: pygeoapi collection id holding the observations, e.g. ``"climate-hourly"``.
    collection: str = ""
    #: Collection holding the station catalogue, if the service publishes one.
    station_collection: str = ""
    #: Property naming the station in the *observation* collection.
    station_id_field: str = ""
    #: Property naming the station in the *station catalogue* collection.
    catalogue_id_field: str = ""
    #: Properties to read the station's human-readable name from, in order of preference.
    #: Collections disagree: the climate ones use ``STATION_NAME``, AHCCD publishes a bilingual
    #: ``station_name__nom_station``.
    name_fields: tuple[str, ...] = ("STATION_NAME", "name")
    #: Property holding the observation time.
    time_field: str = ""
    #: Properties that identify or time-stamp a row rather than measure something.
    skip_fields: frozenset[str] = frozenset()
    #: Suffix marking a QC/flag sibling of a measurement property.
    qc_suffix: str = "_FLAG"
    #: Rough samples per day, used for the row estimate shown in the Catalog.
    samples_per_day: float = 24.0
    #: pandas **Period** alias ("D", "M", "Y") for collections whose rows summarize a period
    #: and are labelled by the period's first instant.
    #:
    #: Without it, asking for "15 July noon to 17 July" silently returns fewer days than
    #: "15 July to 17 July": the 15 July summary is stamped 00:00Z, so the trim discards it even
    #: though the day it describes overlaps the request. A period belongs to a window when its
    #: *interval* overlaps, not when its label instant happens to fall inside.
    period: str | None = None
    #: Reject stations whose period of record for *this* dataset is entirely absent. The ECCC
    #: station catalogue lists every station once and marks the datasets it lacks with null
    #: dates, so without this a query finds ~110 "stations" of which only a handful hold hourly
    #: data, and then makes a pointless request to each one.
    require_record_period: bool = False

    # ------------------------------------------------------------------ urls

    @property
    def items_url(self) -> str:
        return self.provider.collection_url(self.collection)

    @property
    def stations_url(self) -> str:
        return self.provider.collection_url(self.station_collection)

    # ------------------------------------------------------------------ discovery

    def discover(self, query: Query) -> list[StationMatch]:
        if not self.wants_anything(query):
            return []
        if not self.station_collection:
            return self.discover_from_data(query)

        matches: list[StationMatch] = []
        params = {"bbox": _bbox_param(query)}
        for feature in paginate_ogc_items(
            self.stations_url,
            params,
            provider=self.name,
            max_items=int(query.option("max_items", 250_000)),
        ):
            match = self.station_from_feature(query, feature)
            if match is not None:
                matches.append(match.attach_site(query))
        log.debug("%s discovered %d station(s)", self.name, len(matches))
        return matches

    def discover_from_data(self, query: Query) -> list[StationMatch]:
        """Fallback for collections with no station catalogue: sample the data itself."""
        raise NotImplementedError(
            f"{self.name} has no station catalogue and no discover_from_data() override"
        )

    def station_from_feature(
        self, query: Query, feature: Mapping[str, Any]
    ) -> StationMatch | None:
        """Turn one station-catalogue feature into a match, or ``None`` to reject it."""
        point = point_from_feature(feature)
        if point is None:
            return None
        lat, lon = point
        if not query.contains(lat, lon):
            return None

        props = feature.get("properties") or {}
        station_id = props.get(self.catalogue_id_field or self.station_id_field)
        if station_id in (None, ""):
            return None

        first, last = self.record_period(props)
        if self.require_record_period and first is None and last is None:
            return None
        if not query.overlaps(first, last):
            return None

        return self.new_match(
            station_id=str(station_id),
            name=self.station_name(props),
            lat=lat,
            lon=lon,
            variables=tuple(sorted(self.variables)),
            n_rows_est=int(query.days * self.samples_per_day),
            first=_maybe_ts(first),
            last=_maybe_ts(last),
            extra={"properties": dict(props)},
        )

    def station_name(self, props: Mapping[str, Any]) -> str:
        """The station's display name, from the first of :attr:`name_fields` that has one."""
        for field_name in self.name_fields:
            value = props.get(field_name)
            if value:
                return str(value)
        return ""

    def record_period(self, props: Mapping[str, Any]) -> tuple[Any, Any]:
        """The station's period of record, used to skip stations that cannot cover the window."""
        return None, None

    # ------------------------------------------------------------------ retrieval

    def fetch(self, query: Query, matches: list[StationMatch]) -> list[StationSeries]:
        results = map_threads(
            lambda m: self.fetch_station(query, m),
            matches,
            max_workers=int(query.option("max_workers", DEFAULT_MAX_WORKERS)),
            label=f"{self.name} station",
        )
        return [r for r in results if r is not None]

    def fetch_station(self, query: Query, match: StationMatch) -> StationSeries | None:
        params = dict(self.station_filter(match))
        params["datetime"] = self.datetime_param(query)

        features = list(
            paginate_ogc_items(
                self.items_url,
                params,
                provider=self.name,
                max_items=int(query.option("max_items", 250_000)),
            )
        )
        rows = [f.get("properties") or {} for f in features]
        return self.series_from_rows(query, match, rows, features)

    def station_filter(self, match: StationMatch) -> dict[str, Any]:
        """Query parameters that isolate one station in the observation collection."""
        return {self.station_id_field: match.station_id}

    def datetime_param(self, query: Query) -> str:
        return query.interval_iso

    # ------------------------------------------------------------------ shaping

    def series_from_rows(
        self,
        query: Query,
        match: StationMatch,
        rows: list[Mapping[str, Any]],
        features: list[Mapping[str, Any]] | None = None,
    ) -> StationSeries | None:
        """Turn raw GeoJSON properties into a CF-described, time-indexed series."""
        available = _ordered_keys(rows)
        # Deliberately not filtered by query.variables. The response already carries every
        # property, so dropping columns here would discard data that has already crossed the
        # network and cost nothing to keep. `variables=` selects which sources and stations to
        # fetch; it is not a projection over what they returned.
        specs = cf.resolve_fields(
            self.fields,
            available,
            include_unmapped=self.include_unmapped(query),
            skip=self.effective_skip(),
            is_qc=self.is_qc_field,
            units_for=lambda raw: self.units_for(raw, rows),
        )

        to_cf = self.to_cf_units(query)
        records: list[dict[str, Any]] = []
        for raw_row in rows:
            row = self.clean_row(raw_row)
            time_value = self.extract_time(row)
            if time_value is None:
                continue
            record: dict[str, Any] = {"time": time_value}
            for raw, spec in specs.items():
                record[spec.var] = cf.convert(row.get(raw), spec, to_cf_units=to_cf)
                qc_raw = self.qc_field_for(raw, spec)
                if qc_raw and qc_raw in row and row.get(qc_raw) is not None:
                    record[f"{spec.var}_qc"] = row.get(qc_raw)
            records.append(record)

        frame = frame_from_records(records)
        frame = trim_to_window(frame, *self.trim_window(query))
        frame = drop_orphan_qc(frame)
        var_attrs: dict[str, dict[str, Any]] = {}
        for raw, spec in specs.items():
            if spec.var in frame.columns:
                var_attrs[spec.var] = cf.cf_attrs(spec, units=spec.units, to_cf_units=to_cf)
            qc_col = f"{spec.var}_qc"
            if qc_col in frame.columns:
                var_attrs[qc_col] = {
                    "long_name": f"quality flag for {spec.var}",
                    "source_field": self.qc_field_for(raw, spec) or "",
                }

        return StationSeries(
            match=match,
            frame=frame,
            node_path=f"{self.node_path}/{_safe(match.station_id)}",
            attrs=self.node_attrs(query, match),
            var_attrs=var_attrs,
        )

    def node_attrs(self, query: Query, match: StationMatch) -> dict[str, Any]:
        return self.base_attrs(
            title=f"{match.name} ({match.station_id}) — {self.title or self.name}",
            source_url=f"{self.items_url}?{self.station_id_field}={match.station_id}",
            collection=self.collection,
            station_id=match.station_id,
            site=match.site,
        )

    # ------------------------------------------------------------------ per-source hooks

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

    def clean_row(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        """Normalize one raw record before it is shaped.

        The default is a pass-through. Override it for collections that spell "missing" as a
        sentinel value rather than ``null`` — AHCCD writes ``-9999.9``, and left alone that
        becomes a real number in a mean.
        """
        return row

    def extract_time(self, row: Mapping[str, Any]) -> Any:
        return row.get(self.time_field)

    def effective_skip(self) -> frozenset[str]:
        return self.skip_fields | {self.time_field, self.station_id_field}

    def is_qc_field(self, raw: str) -> bool:
        return bool(self.qc_suffix) and raw.endswith(self.qc_suffix)

    def qc_field_for(self, raw: str, spec: cf.FieldSpec) -> str | None:
        if spec.qc_field:
            return spec.qc_field
        return f"{raw}{self.qc_suffix}" if self.qc_suffix else None

    def units_for(self, raw: str, rows: list[Mapping[str, Any]]) -> str | None:
        """Units published alongside the value, for services that provide them per record."""
        return None


# --------------------------------------------------------------------------- helpers


def _bbox_param(query: Query) -> str:
    bbox = query.bbox or (-180.0, -90.0, 180.0, 90.0)
    return ",".join(f"{v:.6f}" for v in bbox)


def _ordered_keys(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Every property name seen, in first-seen order (rows are not always uniform)."""
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    return list(seen)


def _safe(text: Any) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(text)) or "unknown"


def _maybe_ts(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:  # noqa: BLE001 - a malformed catalogue date is not fatal
        return None
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
