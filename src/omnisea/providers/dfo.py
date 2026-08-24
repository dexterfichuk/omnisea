"""Fisheries and Oceans Canada / Canadian Hydrographic Service — the IWLS water level API.

``https://api-iwls.dfo-mpo.gc.ca/api/v1``

Two upstream behaviours shape this adapter, both verified against the live service:

* **The station list has no bbox filter.** ``GET /stations`` returns all ~1573 stations, so
  spatial selection happens client-side over a cached copy of that list.
* **The window limit depends on resolution.** ``ONE_MINUTE`` requests are capped at 7 days;
  every coarser resolution is capped at 31 days. Requests are chunked to whichever limit applies
  rather than to a single conservative one — chunking a month of hourly data into five
  unnecessary requests would be four round-trips of pure waste.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any

from .. import cf
from ..errors import QueryError, UpstreamError
from ..http import DEFAULT_MAX_WORKERS, chunk_time, get_json, map_threads
from ..query import Query
from .base import Provider, RetrievalSource, StationMatch, StationSeries, frame_from_records

log = logging.getLogger("omnisea.dfo")

__all__ = ["DfoProvider", "clear_cache"]

BASE = "https://api-iwls.dfo-mpo.gc.ca/api/v1"

#: Minutes between samples for each IWLS resolution.
RESOLUTION_MINUTES: dict[str, int] = {
    "ONE_MINUTE": 1,
    "THREE_MINUTES": 3,
    "FIVE_MINUTES": 5,
    "FIFTEEN_MINUTES": 15,
    "SIXTY_MINUTES": 60,
}

#: Confirmed live: 7 days at one-minute resolution, 31 days otherwise.
MAX_DAYS_ONE_MINUTE = 7
MAX_DAYS_COARSE = 31

#: One-minute data is 10,080 rows per station per week, which is rarely what someone wants from
#: a multi-station area query. Override with ``resolution="ONE_MINUTE"``.
DEFAULT_RESOLUTION = "FIFTEEN_MINUTES"

#: ``wlp`` (the full harmonic prediction series) doubles the payload, so it is opt-in.
DEFAULT_SERIES = ("wlo", "wlp-hilo")

SERIES_NODES = {
    "wlo": "in_situ/tides",
    "wlp": "predictions/tides",
    "wlp-hilo": "predictions/tides_hilo",
}

_stations_cache: list[dict[str, Any]] | None = None
_metadata_cache: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def clear_cache() -> None:
    """Drop the cached station list and metadata (used by tests)."""
    global _stations_cache
    with _lock:
        _stations_cache = None
        _metadata_cache.clear()


class DfoProvider(Provider):
    name = "dfo"
    title = "Fisheries and Oceans Canada / Canadian Hydrographic Service"
    base_url = BASE
    license = "Fisheries and Oceans Canada — Open Government Licence – Canada"
    terms_url = "https://open.canada.ca/en/open-government-licence-canada"

    def build_sources(self) -> Sequence[RetrievalSource]:
        return [DfoTidesSource(self)]

    # ------------------------------------------------------------------ cached lookups

    def all_stations(self) -> list[dict[str, Any]]:
        """The full IWLS station list, fetched once per process."""
        global _stations_cache
        with _lock:
            if _stations_cache is None:
                log.debug("fetching IWLS station list")
                _stations_cache = self.get_json("stations", source="dfo_tides")
            return _stations_cache

    def station_metadata(self, station_id: str) -> dict[str, Any]:
        """Per-station metadata (datums, region, heights), memoized."""
        with _lock:
            cached = _metadata_cache.get(station_id)
        if cached is not None:
            return cached
        try:
            meta = self.get_json(f"stations/{station_id}/metadata", source="dfo_tides")
        except UpstreamError:
            # Metadata is enrichment, not measurement; a station without it still returns data.
            log.warning("no IWLS metadata for station %s", station_id, exc_info=True)
            meta = {}
        with _lock:
            _metadata_cache[station_id] = meta
        return meta


class DfoTidesSource(RetrievalSource):
    """Water levels: observed, predicted, and predicted high/low events.

    Observations and predictions land in **separate branches** of the tree
    (``in_situ/tides`` vs ``predictions/tides``) so that a harmonic prediction can never be
    mistaken for a measurement — the two look identical in a flat table and are not remotely
    the same thing.
    """

    name = "dfo_tides"
    title = "DFO IWLS water levels"
    node_path = "in_situ/tides"
    feature_type = "timeSeries"

    fields = {
        "wlo": cf.FieldSpec(
            var="water_surface_height_above_reference_datum",
            standard_name="water_surface_height_above_reference_datum",
            units="m",
            long_name="Observed water level above chart datum",
            qc_field="qcFlagCode",
            comment="CHS official observed water level (IWLS series 'wlo').",
        ),
        "wlp": cf.FieldSpec(
            var="water_surface_height_above_reference_datum_predicted",
            standard_name="water_surface_height_above_reference_datum",
            units="m",
            long_name="Predicted (harmonic) water level above chart datum",
            comment=(
                "Harmonic tide prediction (IWLS series 'wlp'), kept in a separate node from the "
                "observations so predictions can never be mistaken for measurements."
            ),
        ),
        "wlp-hilo": cf.FieldSpec(
            var="water_surface_height_above_reference_datum_at_extremum",
            standard_name="water_surface_height_above_reference_datum",
            units="m",
            long_name="Predicted high/low tide height above chart datum",
            comment=(
                "Predicted tidal extrema (IWLS series 'wlp-hilo'); the time axis is the "
                "irregular series of turning points, not a regular grid."
            ),
        ),
    }

    # ------------------------------------------------------------------ discovery

    def discover(self, query: Query) -> list[StationMatch]:
        series_wanted = self._series_for(query)
        if not series_wanted:
            return []

        matches: list[StationMatch] = []
        for station in self.provider.all_stations():
            lat, lon = station.get("latitude"), station.get("longitude")
            if lat is None or lon is None:
                continue
            if not query.contains(float(lat), float(lon)):
                continue

            available = {ts.get("code") for ts in station.get("timeSeries") or []}
            usable = [code for code in series_wanted if code in available]
            if not usable:
                continue

            match = self.new_match(
                station_id=str(station.get("code") or station.get("id")),
                name=str(station.get("officialName") or ""),
                lat=float(lat),
                lon=float(lon),
                variables=("water_surface_height_above_reference_datum",),
                n_rows_est=self._estimate_rows(query, usable),
                extra={
                    "iwls_id": station.get("id"),
                    "series": usable,
                    "operating": station.get("operating"),
                    "station_type": station.get("type"),
                },
            )
            matches.append(match.attach_site(query))

        log.debug("dfo_tides discovered %d station(s)", len(matches))
        return matches

    # ------------------------------------------------------------------ retrieval

    def fetch(self, query: Query, matches: list[StationMatch]) -> list[StationSeries]:
        jobs: list[tuple[StationMatch, str]] = []
        for match in matches:
            for code in match.extra.get("series") or self._series_for(query):
                jobs.append((match, code))

        results = map_threads(
            lambda job: self._fetch_series(query, *job),
            jobs,
            max_workers=int(query.option("max_workers", DEFAULT_MAX_WORKERS)),
            label="iwls series",
        )
        return [r for r in results if r is not None]

    def _fetch_series(
        self, query: Query, match: StationMatch, code: str
    ) -> StationSeries | None:
        # The IWLS internal id comes from discovery; a station vanishing from the tree because
        # it was missing would be far worse than a clear error.
        station_id = match.require("iwls_id")

        resolution = self._resolution(query)
        # wlp-hilo is an event series: irregular by nature, and it takes no resolution parameter.
        use_resolution = code != "wlp-hilo"
        max_days = (
            MAX_DAYS_ONE_MINUTE
            if (use_resolution and resolution == "ONE_MINUTE")
            else MAX_DAYS_COARSE
        )

        rows: list[dict[str, Any]] = []
        for start, end in chunk_time(query.start, query.end, max_days=max_days):
            params: dict[str, Any] = {
                "time-series-code": code,
                "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            if use_resolution:
                params["resolution"] = resolution
            payload = get_json(
                f"{BASE}/stations/{station_id}/data", params, provider=self.name
            )
            if isinstance(payload, list):
                rows.extend(payload)

        spec = self.fields[code]
        include_unmapped = self.include_unmapped(query)
        to_cf = self.to_cf_units(query)

        records: list[dict[str, Any]] = []
        for row in rows:
            record: dict[str, Any] = {
                "time": row.get("eventDate"),
                spec.var: cf.convert(row.get("value"), spec, to_cf_units=to_cf),
                f"{spec.var}_qc": row.get("qcFlagCode"),
            }
            if include_unmapped:
                # No CF equivalent, but it says whether a human verified the value.
                record["reviewed"] = row.get("reviewed")
            records.append(record)

        frame = frame_from_records(records)
        var_attrs: dict[str, dict[str, Any]] = {
            spec.var: cf.cf_attrs(spec, to_cf_units=to_cf),
            f"{spec.var}_qc": {
                "long_name": "IWLS QC flag code",
                "comment": "As published by IWLS; 1 indicates a verified observation.",
                "source_field": "qcFlagCode",
            },
        }
        if include_unmapped:
            var_attrs["reviewed"] = {
                "long_name": "reviewed",
                "comment": "IWLS 'reviewed' flag, carried through unmapped (no CF equivalent).",
                cf.MAPPED_ATTR: 0,
                "source_field": "reviewed",
            }

        attrs = self._node_attrs(query, match, code, station_id, resolution, use_resolution)
        return StationSeries(
            match=match,
            frame=frame,
            node_path=f"{SERIES_NODES[code]}/{match.station_id}",
            attrs=attrs,
            var_attrs=var_attrs,
        )

    def _node_attrs(
        self,
        query: Query,
        match: StationMatch,
        code: str,
        station_id: str,
        resolution: str,
        use_resolution: bool,
    ) -> dict[str, Any]:
        meta = self.provider.station_metadata(station_id)
        attrs = self.base_attrs(
            title=f"{match.name} ({match.station_id}) — IWLS {code}",
            source_url=f"{BASE}/stations/{station_id}/data?time-series-code={code}",
            station_id=match.station_id,
            iwls_station_id=station_id,
            iwls_series_code=code,
            datum="chart datum (CD)",
            chs_region_code=meta.get("chsRegionCode"),
            is_tidal=meta.get("isTidal"),
            site=match.site,
            summary="Observed water level" if code == "wlo" else "Predicted water level",
        )
        datums = {
            str(d.get("code")): d.get("offset")
            for d in meta.get("datums") or []
            if d.get("code") is not None
        }
        for datum_code, offset in datums.items():
            attrs[f"datum_offset_{datum_code}"] = offset
        if datums:
            attrs["datum_offset_comment"] = (
                "Offsets convert chart datum to the named vertical datum "
                "(value_in_datum = value + offset)."
            )
        if use_resolution:
            attrs["resolution"] = resolution
        return attrs

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _resolution(query: Query) -> str:
        resolution = str(query.option("resolution", DEFAULT_RESOLUTION)).upper()
        if resolution not in RESOLUTION_MINUTES:
            raise QueryError(
                f"unknown IWLS resolution {resolution!r}; "
                f"choose one of {', '.join(RESOLUTION_MINUTES)}"
            )
        return resolution

    @staticmethod
    def _series_for(query: Query) -> list[str]:
        requested = query.option("series")
        if requested:
            codes = [requested] if isinstance(requested, str) else list(requested)
            unknown = [c for c in codes if c not in SERIES_NODES]
            if unknown:
                raise QueryError(
                    f"unknown IWLS series {unknown}; choose from {sorted(SERIES_NODES)}"
                )
            return codes
        if not query.wants(
            "water_surface_height_above_reference_datum",
            "sea_surface_height_above_reference_datum",
        ):
            return []
        return list(DEFAULT_SERIES)

    def _estimate_rows(self, query: Query, codes: list[str]) -> int:
        minutes = RESOLUTION_MINUTES[self._resolution(query)]
        per_day = 24 * 60 / minutes
        total = 0
        for code in codes:
            # Tidal extrema are roughly four turning points a day, not a regular series.
            total += int(query.days * (4 if code == "wlp-hilo" else per_day))
        return total
