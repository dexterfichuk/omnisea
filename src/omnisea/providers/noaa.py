"""NOAA CO-OPS — US tide gauges, natively, with the datum stated.

``https://api.tidesandcurrents.noaa.gov``

CO-OPS water levels are reachable through the IOOS ERDDAP mirror, but the mirror renames the
variable to ``sea_surface_height_above_sea_level`` while the numbers are above **MLLW**, and it
serves no predictions and no datum choice. This adapter talks to CO-OPS directly, mirrors
``dfo_tides``' branch layout exactly — observations under ``in_situ/tides``, predicted extrema
under ``predictions/tides_hilo`` — and stamps the datum on the node and the variable, so a tree
holding a Canadian gauge on chart datum beside an American one on MLLW says so where the
columns meet.

Two upstream behaviours shape the code, both verified live:

* **Six-minute data is capped at 31 days per request**; hi/lo predictions allow a year.
  Requests are chunked to whichever limit applies.
* **"No data" is an error payload**, ``{"error": {"message": "No data was found..."}}`` with
  HTTP 200. That is an empty result, not a failure; any other error message still raises.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import pandas as pd

from .. import cf
from ..errors import UpstreamError
from ..http import DEFAULT_MAX_WORKERS, NEVER_CACHE, chunk_time, get_json, map_threads
from ..query import Query, register_option
from .base import Provider, RetrievalSource, StationMatch, StationSeries, frame_from_records

log = logging.getLogger("omnisea.noaa")

register_option(
    "coops_datum",
    "noaa_coops: vertical datum for water levels — MLLW (default), MSL, MHW, NAVD, STND; "
    "Great Lakes stations default to IGLD",
)
register_option(
    "coops_high_low",
    "noaa_coops: True also fetches the OBSERVED tidal extrema (product high_low) into "
    "in_situ/tides_extrema — distinct from the predicted extrema despite the similar name",
)

__all__ = ["CoopsProvider", "clear_cache"]

API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
STATIONS = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"

#: Verified live: 31 days of six-minute data per request; a year of hi/lo predictions.
MAX_DAYS_WATER_LEVEL = 31
MAX_DAYS_HILO = 365

#: Six-minute water_level begins here; before it, only the hourly_height archive exists.
#: A request that ignores this simply fails for older windows — the gap a GNSS reflectometry
#: project hit without ever learning why.
SIX_MINUTE_ERA = pd.Timestamp("1996-01-01", tz="UTC")

SERIES_NODES = {
    "water_level": "in_situ/tides",
    "hilo": "predictions/tides_hilo",
    # Observed extrema are measurements and live beside the observations — the product name
    # ("high_low") reads like the prediction product, which is exactly the confusion the
    # branch split exists to prevent.
    "high_low": "in_situ/tides_extrema",
}

_stations_cache: list[dict[str, Any]] | None = None
_lock = threading.Lock()


def clear_cache() -> None:
    """Drop the cached station list (used by tests)."""
    global _stations_cache
    with _lock:
        _stations_cache = None


class CoopsProvider(Provider):
    name = "noaa_coops"
    title = "NOAA Center for Operational Oceanographic Products and Services"
    base_url = "https://api.tidesandcurrents.noaa.gov"
    license = "US Government work — public domain (NOAA CO-OPS)"
    terms_url = "https://tidesandcurrents.noaa.gov/disclaimers.html"

    #: Water levels are minutes old and a stale one is a wrong number; the station catalogue
    #: is ~300 entries that change a few times a year.
    cache_policy = {
        "api.tidesandcurrents.noaa.gov/api/prod/datagetter*": NEVER_CACHE,
        "api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json*": timedelta(days=7),
        # The datums ladder changes with the tidal epoch — decades apart.
        "api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/*": timedelta(days=7),
    }

    def clear_cache(self) -> None:
        clear_cache()

    def build_sources(self) -> Sequence[RetrievalSource]:
        return [CoopsWaterSource(self)]

    def datum_ladder(self, station_id: str) -> dict[str, Any]:
        """Datum offsets and epoch for one station, as flat node attributes."""
        try:
            payload = self.get_json(
                f"mdapi/prod/webapi/stations/{station_id}/datums.json",
                params={"units": "metric"},  # the mdapi default is feet
            )
        except UpstreamError:
            log.debug("no datums ladder for CO-OPS %s", station_id, exc_info=True)
            return {}
        attrs: dict[str, Any] = {}
        if payload.get("epoch"):
            attrs["datum_epoch"] = str(payload["epoch"])
        if payload.get("OrthometricDatum"):
            attrs["orthometric_datum"] = str(payload["OrthometricDatum"])
        for entry in payload.get("datums") or []:
            name, value = entry.get("name"), entry.get("value")
            if name and value is not None:
                attrs[f"datum_offset_{name}"] = float(value)
        if attrs:
            attrs["datum_offsets_units"] = "m"
        return attrs

    def all_stations(self) -> list[dict[str, Any]]:
        """The CO-OPS water-level station list, fetched once per process.

        The list has no bbox filter, so spatial selection happens client-side — the same shape
        as the IWLS list on the Canadian side.
        """
        global _stations_cache
        with _lock:
            if _stations_cache is None:
                log.debug("fetching CO-OPS station list")
                payload = self.get_json(
                    "mdapi/prod/webapi/stations.json",
                    params={"type": "waterlevels"},
                    source="noaa_coops",
                )
                _stations_cache = list(payload.get("stations") or [])
            return _stations_cache


class CoopsWaterSource(RetrievalSource):
    """US water levels: six-minute observations and predicted high/low events.

    Branch-for-branch symmetric with ``dfo_tides``, so a cross-border query lands both
    countries' gauges in the same tree shape — observations under ``in_situ/tides``,
    predictions under ``predictions/tides_hilo``, never mixed.
    """

    name = "noaa_coops"
    title = "NOAA CO-OPS water levels"
    node_path = "in_situ/tides"
    feature_type = "timeSeries"
    #: Six-minute water levels.
    samples_per_day = 240.0

    fields = {
        "water_level": cf.FieldSpec(
            var="water_surface_height_above_reference_datum",
            standard_name="water_surface_height_above_reference_datum",
            units="m",
            long_name="Observed water level above station datum",
            qc_field="q",
            comment=(
                "CO-OPS six-minute observed water level. The reference datum is stated in "
                "the vertical_datum attribute — MLLW unless coops_datum= chose another."
            ),
        ),
        "high_low": cf.FieldSpec(
            var="water_surface_height_above_reference_datum_at_verified_extremum",
            standard_name="water_surface_height_above_reference_datum",
            units="m",
            long_name="Observed high/low tide height above station datum",
            comment=(
                "CO-OPS verified OBSERVED tidal extrema — measurements, despite sharing a "
                "shape with the predicted product; kept under in_situ for exactly that "
                "reason."
            ),
        ),
        "hilo": cf.FieldSpec(
            var="water_surface_height_above_reference_datum_at_extremum",
            standard_name="water_surface_height_above_reference_datum",
            units="m",
            long_name="Predicted high/low tide height above station datum",
            comment=(
                "CO-OPS predicted tidal extrema; the time axis is the irregular series of "
                "turning points, not a regular grid. Kept in a separate node from the "
                "observations so predictions can never be mistaken for measurements."
            ),
        ),
    }

    def locate(self, station_id: str) -> tuple[float, float, str] | None:
        for st in self.provider.all_stations():
            if str(st.get("id")) == str(station_id):
                return float(st["lat"]), float(st["lng"]), str(st.get("name") or station_id)
        return None

    # ------------------------------------------------------------------ discovery

    def discover(self, query: Query) -> list[StationMatch]:
        matches: list[StationMatch] = []
        for station in self.provider.all_stations():
            lat, lon = station.get("lat"), station.get("lng")
            if lat is None or lon is None:
                continue
            if not query.contains(float(lat), float(lon)):
                continue
            matches.append(
                self.new_match(
                    station_id=str(station.get("id")),
                    name=str(station.get("name") or ""),
                    lat=float(lat),
                    lon=float(lon),
                    variables=("water_surface_height_above_reference_datum",),
                    n_rows_est=self.row_estimate(query),
                    extra={
                        "state": station.get("state"),
                        # Great Lakes gauges publish against IGLD, and asking them for MLLW
                        # is an upstream error, not an empty result.
                        "greatlakes": bool(station.get("greatlakes")),
                    },
                ).attach_site(query)
            )
        log.debug("noaa_coops discovered %d station(s)", len(matches))
        return matches

    # ------------------------------------------------------------------ retrieval

    def fetch(self, query: Query, matches: list[StationMatch]) -> list[StationSeries]:
        codes = ["water_level", "hilo"]
        if query.option("coops_high_low"):
            codes.append("high_low")
        jobs = [(match, code) for match in matches for code in codes]
        results = map_threads(
            lambda job: self._fetch_series(query, *job),
            jobs,
            max_workers=int(query.option("max_workers", DEFAULT_MAX_WORKERS)),
            label="coops series",
        )
        return [r for r in results if r is not None]

    def _requests(self, query: Query, code: str):
        """(product, start, end) chunks for one series, splitting eras for observations.

        Six-minute ``water_level`` exists from 1996; the decades before it live only in
        ``hourly_height``. One column serves both eras — same quantity, same datum, hourly
        rows simply spaced wider — instead of the pre-1996 request failing outright.
        """
        if code == "hilo":
            for start, end in chunk_time(query.start, query.end, max_days=MAX_DAYS_HILO):
                yield "predictions", start, end
            return
        if code == "high_low":
            for start, end in chunk_time(query.start, query.end, max_days=MAX_DAYS_HILO):
                yield "high_low", start, end
            return
        if query.start < SIX_MINUTE_ERA:
            archive_end = min(query.end, SIX_MINUTE_ERA)
            for start, end in chunk_time(query.start, archive_end, max_days=MAX_DAYS_HILO):
                yield "hourly_height", start, end
        if query.end > SIX_MINUTE_ERA:
            modern_start = max(query.start, SIX_MINUTE_ERA)
            for start, end in chunk_time(
                modern_start, query.end, max_days=MAX_DAYS_WATER_LEVEL
            ):
                yield "water_level", start, end

    def _datum(self, query: Query, match: StationMatch) -> str:
        chosen = str(query.option("coops_datum") or "").upper()
        if chosen:
            return chosen
        return "IGLD" if match.extra.get("greatlakes") else "MLLW"

    def _fetch_series(
        self, query: Query, match: StationMatch, code: str
    ) -> StationSeries | None:
        datum = self._datum(query, match)
        rows: list[dict[str, Any]] = []
        for product, start, end in self._requests(query, code):
            params: dict[str, Any] = {
                "product": product,
                "station": match.station_id,
                "datum": datum,
                "units": "metric",
                "time_zone": "gmt",
                "format": "json",
                "application": "omnisea",
                "begin_date": start.strftime("%Y%m%d %H:%M"),
                "end_date": end.strftime("%Y%m%d %H:%M"),
            }
            if code == "hilo":
                params["interval"] = "hilo"
            if product == "hourly_height":
                # The archive era: hourly means, one column with the six-minute rows. The
                # cadence change is visible in the data and stated in the comment; the
                # alternative was a request that fails for any window before 1996.
                params.pop("interval", None)
            payload = get_json(API, params, provider=self.name)
            error = str((payload.get("error") or {}).get("message") or "")
            if error:
                # "No data was found" is an answer, not a failure — the station simply has
                # nothing for the window (or, for predictions, is a subordinate station).
                if "no data" in error.lower() or "not a valid datum" in error.lower():
                    log.debug("coops %s %s: %s", match.station_id, code, error.strip())
                    continue
                raise UpstreamError(error.strip(), provider=self.name, url=API)
            rows.extend(payload.get("predictions" if product == "predictions" else "data") or [])

        spec = self.fields[code]
        include_unmapped = self.include_unmapped(query)
        to_cf = self.to_cf_units(query)

        records: list[dict[str, Any]] = []
        for row in rows:
            record: dict[str, Any] = {
                # time_zone=gmt: the stamp is UTC wearing no marker; parsed utc=True below.
                "time": row.get("t"),
                spec.var: cf.convert(_number(row.get("v")), spec, to_cf_units=to_cf),
            }
            if code == "water_level":
                if row.get("q") is not None:
                    record[f"{spec.var}_qc"] = row.get("q")
                if include_unmapped and row.get("s") is not None:
                    # One-sigma standard deviation of the six-minute samples — no CF name,
                    # but it is the published measurement uncertainty.
                    record["sigma"] = _number(row.get("s"))
            elif include_unmapped and row.get("ty") is not None:
                record["extremum_type"] = row.get("ty")
            elif include_unmapped and row.get("type") is not None:
                record["extremum_type"] = row.get("type")
            records.append(record)

        frame = frame_from_records(records)
        var_attrs: dict[str, dict[str, Any]] = {
            spec.var: {**cf.cf_attrs(spec, to_cf_units=to_cf), "vertical_datum": datum},
        }
        if code == "water_level":
            var_attrs[f"{spec.var}_qc"] = {
                "long_name": "CO-OPS quality flag",
                "comment": "As published: 'v' verified, 'p' preliminary.",
                "source_field": "q",
            }
            if include_unmapped:
                var_attrs["sigma"] = {
                    "long_name": "standard deviation of six-minute water level samples",
                    "units": "m",
                    cf.MAPPED_ATTR: 0,
                    "source_field": "s",
                }
        elif include_unmapped:
            var_attrs["extremum_type"] = {
                "long_name": "predicted extremum type",
                "comment": "H/HH high, L/LL low, as published by CO-OPS.",
                cf.MAPPED_ATTR: 0,
                "source_field": "type",
            }

        attrs = self.base_attrs(
            source_url=(
                f"{API}?product={'predictions&interval=hilo' if code == 'hilo' else 'water_level'}"
                f"&station={match.station_id}&datum={datum}&units=metric&time_zone=gmt"
            ),
            datum=datum,
            station_state=str(match.extra.get("state") or "") or None,
        )
        if code == "water_level":
            # The full datums ladder turns "the datum is stated" into "the datum is
            # convertible": two US gauges both on MLLW are not on the same surface, and the
            # offsets are what relate them. Enrichment, not measurement — a station without
            # a published ladder still returns its data.
            attrs.update(self.provider.datum_ladder(match.station_id))
        return StationSeries(
            match=match,
            frame=frame,
            node_path=f"{SERIES_NODES[code]}/{match.station_id}",
            attrs=attrs,
            var_attrs=var_attrs,
        )


def _number(value: Any) -> float | None:
    """CO-OPS sends numbers as strings and gaps as empty strings."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
