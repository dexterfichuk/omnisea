"""USGS NWIS — US river gauges, the partner to ECCC's hydrometric collections.

``https://waterservices.usgs.gov/nwis``

Discovery uses the site service with ``seriesCatalogOutput=true``, which states each site's
period of record **per parameter** — so a gauge discontinued in 1972 excludes itself from a
2024 query by its own dates rather than being discovered, fetched and returned empty.
Retrieval uses the instantaneous-values service, which serves the archived record at the
gauge's native cadence (typically 15 minutes).

Values arrive in the units USGS publishes — cubic feet per second and feet — and stay that
way, with the units recorded beside them; ``to_cf_units=True`` converts to m³/s and metres.
Nodes land under the same ``in_situ/hydrometric`` branch as ECCC's gauges, so a cross-border
river query produces one tree shape. Station identifiers cannot collide: USGS ids are numeric,
ECCC's are alphanumeric.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from datetime import timedelta
from typing import Any

import pandas as pd

from .. import cf
from ..errors import UpstreamError
from ..http import (
    DEFAULT_MAX_WORKERS,
    NEVER_CACHE,
    chunk_time,
    get_json,
    get_text,
    map_threads,
)
from ..query import Query
from .base import Provider, RetrievalSource, StationMatch, StationSeries, frame_from_records

log = logging.getLogger("omnisea.usgs")

__all__ = ["UsgsProvider"]

SITE_SERVICE = "https://waterservices.usgs.gov/nwis/site/"
IV_SERVICE = "https://waterservices.usgs.gov/nwis/iv/"
DV_SERVICE = "https://waterservices.usgs.gov/nwis/dv/"

#: Typical NWIS reporting interval is 15 minutes.
SAMPLES_PER_DAY = 96.0

#: Kind to the service: a decade of 15-minute data in one request is a heavy ask.
MAX_DAYS_PER_REQUEST = 120


class UsgsProvider(Provider):
    name = "usgs"
    title = "US Geological Survey / National Water Information System"
    base_url = "https://waterservices.usgs.gov/nwis"
    license = "US Government work — public domain (USGS)"
    terms_url = "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits"

    cache_policy = {
        "waterservices.usgs.gov/nwis/site*": timedelta(days=7),
        "waterservices.usgs.gov/nwis/iv*": NEVER_CACHE,
        "waterservices.usgs.gov/nwis/dv*": timedelta(hours=1),
    }

    def build_sources(self) -> Sequence[RetrievalSource]:
        return [UsgsWaterSource(self), UsgsWaterDailySource(self)]


class UsgsWaterSource(RetrievalSource):
    """US stream gauges: discharge, stage and water temperature at native cadence."""

    name = "usgs_water"
    title = "USGS NWIS instantaneous values"
    node_path = "in_situ/hydrometric"
    feature_type = "timeSeries"
    #: ``data_type_cd`` values whose records this source can actually serve.
    record_kinds = frozenset({"uv", "iv", "rt"})

    #: Keyed by NWIS parameter code. Same CF names as the ECCC hydrometric sources, so a
    #: cross-border query serves comparable columns under identical names.
    fields = {
        "00060": cf.FieldSpec(
            var="river_discharge",
            standard_name="water_volume_transport_in_river_channel",
            units="ft3 s-1",
            long_name="Streamflow",
            cf_units="m3 s-1",
            cf_scale=0.028316846592,
        ),
        "00065": cf.FieldSpec(
            var="water_surface_height_above_reference_datum",
            standard_name="water_surface_height_above_reference_datum",
            units="ft",
            long_name="Gage height",
            cf_units="m",
            cf_scale=0.3048,
        ),
        "00010": cf.FieldSpec(
            var="water_temperature",
            standard_name="sea_water_temperature",
            units="degC",
            long_name="Water temperature",
            cf_units="K",
            cf_offset=273.15,
        ),
    }

    # ------------------------------------------------------------------ discovery

    def discover(self, query: Query) -> list[StationMatch]:
        box = query.bbox
        if box is None:
            return []
        try:
            payload = get_text(
                SITE_SERVICE,
                {
                    "format": "rdb",
                    # NWIS wants lon-lat order — west,south,east,north — at most 7 decimals.
                    "bBox": ",".join(f"{v:.7f}" for v in box),
                    "parameterCd": ",".join(self.fields),
                    "siteType": "ST",
                    "seriesCatalogOutput": "true",
                },
                provider=self.name,
            )
        except UpstreamError as exc:
            if exc.status == 404:
                # NWIS answers an empty result set with 404, the same convention as ERDDAP.
                # Every query outside the US lands here, and "no US gauges in Canada" is an
                # answer, not a failure.
                return []
            raise
        rows = _parse_rdb(payload)

        # One row per (site, parameter, record); fold to one match per site, keeping the
        # union period of record so a discontinued gauge excludes itself by its own dates.
        # data_type_cd names the record kind — 'uv' instantaneous, 'dv' daily, 'qw' grab
        # samples — and only the kind this source fetches counts as availability: one box on
        # the Olympic Peninsula holds 573 water-quality records that the IV service would
        # answer with nothing.
        by_site: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row.get("parm_cd") not in self.fields:
                continue
            if row.get("data_type_cd") not in self.record_kinds:
                continue
            site = by_site.setdefault(
                str(row["site_no"]),
                {
                    "name": row.get("station_nm", ""),
                    "lat": row.get("dec_lat_va"),
                    "lon": row.get("dec_long_va"),
                    "params": set(),
                    "first": None,
                    "last": None,
                },
            )
            site["params"].add(row["parm_cd"])
            begin, end = row.get("begin_date"), row.get("end_date")
            if begin and (site["first"] is None or begin < site["first"]):
                site["first"] = begin
            if end and (site["last"] is None or end > site["last"]):
                site["last"] = end

        matches: list[StationMatch] = []
        for site_no, info in by_site.items():
            try:
                lat, lon = float(info["lat"]), float(info["lon"])
            except (TypeError, ValueError):
                continue
            if not query.contains(lat, lon):
                continue
            first = pd.Timestamp(info["first"], tz="UTC") if info["first"] else None
            last = pd.Timestamp(info["last"], tz="UTC") if info["last"] else None
            if not query.overlaps(first, last):
                continue
            matches.append(
                self.new_match(
                    station_id=site_no,
                    name=str(info["name"]),
                    lat=lat,
                    lon=lon,
                    variables=tuple(sorted(self.fields[p].var for p in info["params"])),
                    n_rows_est=max(1, int(query.days * SAMPLES_PER_DAY)),
                    first=first,
                    last=last,
                    extra={"params": sorted(info["params"])},
                ).attach_site(query)
            )
        log.debug("usgs_water discovered %d site(s)", len(matches))
        return matches

    # ------------------------------------------------------------------ retrieval

    def fetch(self, query: Query, matches: list[StationMatch]) -> list[StationSeries]:
        results = map_threads(
            lambda match: self._fetch_site(query, match),
            matches,
            max_workers=int(query.option("max_workers", DEFAULT_MAX_WORKERS)),
            label="nwis sites",
        )
        return [r for r in results if r is not None]

    @property
    def _service(self) -> str:
        return IV_SERVICE

    def _request_params(self, match: StationMatch, params_wanted: list[str],
                        start: Any, end: Any) -> dict[str, Any]:
        return {
            "format": "json",
            "sites": match.station_id,
            "parameterCd": ",".join(params_wanted),
            "startDT": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endDT": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def _fetch_site(self, query: Query, match: StationMatch) -> StationSeries | None:
        params_wanted = match.extra.get("params") or list(self.fields)
        to_cf = self.to_cf_units(query)

        # One dict per timestamp, columns merged across parameters — the shape
        # frame_from_records wants.
        by_time: dict[str, dict[str, Any]] = {}
        for start, end in chunk_time(query.start, query.end, max_days=MAX_DAYS_PER_REQUEST):
            try:
                payload = get_json(
                    self._service,
                    self._request_params(match, params_wanted, start, end),
                    provider=self.name,
                )
            except UpstreamError as exc:
                if exc.status == 404:
                    continue  # no rows in this chunk — same 404-means-empty convention
                raise
            for series in (payload.get("value") or {}).get("timeSeries") or []:
                code = str(
                    (series.get("variable") or {}).get("variableCode", [{}])[0].get("value")
                )
                spec = self.fields.get(code)
                if spec is None:
                    continue
                no_data = (series.get("variable") or {}).get("noDataValue")
                for block in series.get("values") or []:
                    for point in block.get("value") or []:
                        stamp = str(point.get("dateTime"))
                        record = by_time.setdefault(stamp, {"time": stamp})
                        value = _number(point.get("value"), no_data)
                        record[spec.var] = cf.convert(value, spec, to_cf_units=to_cf)
                        qualifiers = point.get("qualifiers") or []
                        record[f"{spec.var}_qc"] = ",".join(map(str, qualifiers))

        frame = frame_from_records(list(by_time.values()))
        var_attrs: dict[str, dict[str, Any]] = {}
        for code in params_wanted:
            spec = self.fields.get(code)
            if spec is None or spec.var not in frame.columns:
                continue
            var_attrs[spec.var] = cf.cf_attrs(spec, to_cf_units=to_cf)
            var_attrs[f"{spec.var}_qc"] = {
                "long_name": "NWIS qualifier codes",
                "comment": "As published: A approved, P provisional, e estimated.",
                "source_field": "qualifiers",
            }

        return StationSeries(
            match=match,
            frame=frame,
            node_path=f"{self.node_path}/{match.station_id}",
            attrs=self.base_attrs(
                source_url=(
                    f"{IV_SERVICE}?format=json&sites={match.station_id}"
                    f"&parameterCd={','.join(params_wanted)}"
                ),
            ),
            var_attrs=var_attrs,
        )


class UsgsWaterDailySource(UsgsWaterSource):
    """Daily mean discharge, stage and water temperature — the historical archive.

    The partner to ``eccc_hydrometric_daily``, under the matching branch. NWIS labels each
    daily row by the site's **local calendar date** stamped at midnight with no offset —
    ECCC's convention exactly — so the node records ``time_reference`` and ``align()`` reads
    those stamps in station-local time rather than handing an afternoon sample the next
    day's mean.
    """

    name = "usgs_water_daily"
    title = "USGS NWIS daily values"
    node_path = "in_situ/hydrometric_daily"
    record_kinds = frozenset({"dv"})
    period = "D"
    #: One row per day.
    samples_per_day = 1.0

    fields = {
        code: replace(
            spec,
            cell_methods="time: mean",
            long_name=f"Daily mean {spec.long_name.lower()}",
        )
        for code, spec in UsgsWaterSource.fields.items()
    }

    def discover(self, query: Query) -> list[StationMatch]:
        matches = super().discover(query)
        for match in matches:
            match.n_rows_est = max(1, int(query.days))
        return matches

    def _fetch_site(self, query: Query, match: StationMatch) -> StationSeries | None:
        series = super()._fetch_site(query, match)
        if series is None:
            return None
        series.attrs["time_reference"] = (
            "LOCAL_DATE: daily values are labelled by the site's local calendar date and "
            "stamped at midnight with no UTC offset."
        )
        return series

    @property
    def _service(self) -> str:
        return DV_SERVICE

    def _request_params(self, match: StationMatch, params_wanted: list[str],
                        start: Any, end: Any) -> dict[str, Any]:
        return {
            "format": "json",
            "sites": match.station_id,
            "parameterCd": ",".join(params_wanted),
            "statCd": "00003",  # the daily MEAN; the statistic the cell_methods asserts
            "startDT": start.strftime("%Y-%m-%d"),
            "endDT": end.strftime("%Y-%m-%d"),
        }


def _number(value: Any, no_data: Any) -> float | None:
    """NWIS sends numbers as strings and marks gaps with a sentinel (usually -999999)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if no_data is not None and number == float(no_data):
        return None
    return number


def _parse_rdb(text: str) -> list[dict[str, str]]:
    """USGS tab-delimited RDB: comment lines, a header row, a dtype row, then data."""
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if len(lines) < 3:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"), strict=False)) for line in lines[2:]]
