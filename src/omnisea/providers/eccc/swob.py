"""ECCC Surface Weather Observations (SWOB), realtime."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from ... import cf
from ...http import paginate_ogc_items
from ...query import Query
from ..base import StationMatch
from ..ogc import OgcFeaturesSource, point_from_feature
from .common import CM_TO_M, DEGC_TO_K, HPA_TO_PA, KMH_TO_MS

log = logging.getLogger("omnisea.eccc.swob")

__all__ = ["EcccSwobRealtime"]


_SWOB_SUFFIXES = ("-uom", "-qa", "-value", "-data_flag-code_src")

_SWOB_SKIP = frozenset(
    {
        "dataset",
        "id",
        "url",
        "obs_date_tm",
        "processed_date_tm",
        "lat",
        "long",
        "clim_id",
        "msc_id",
        "stn_nam",
        "tc_id",
        "wmo_synop_id",
        "data_pvdr",
        "stn_elev",
        "date_tm",
    }
)


class EcccSwobRealtime(OgcFeaturesSource):
    """Surface Weather Observations, roughly the last 30 days at minute-to-hourly cadence.

    Units are **read from the data**: every measurement property has a ``-uom`` sibling, so this
    source never hardcodes a unit table and never has to be updated when a station changes
    sensors.

    There is no station catalogue for SWOB, so discovery samples a narrow time slice across the
    query area — every active station reports at least hourly, so one hour of records enumerates
    them without pulling the full collection.
    """

    name = "eccc_swob"
    title = "ECCC surface weather observations (SWOB) realtime"
    node_path = "in_situ/weather_realtime"
    collection = "swob-realtime"
    station_collection = ""  # no catalogue; discovery samples the data
    station_id_field = "msc_id-value"
    time_field = "obs_date_tm"
    skip_fields = _SWOB_SKIP
    qc_suffix = "-qa"
    samples_per_day = 24.0 * 6  # 10-minute reporting is typical

    fields = {
        "air_temp": cf.FieldSpec(
            var="air_temperature", standard_name="air_temperature",
            long_name="Air temperature", **DEGC_TO_K,
        ),
        "dwpt_temp": cf.FieldSpec(
            var="dew_point_temperature", standard_name="dew_point_temperature",
            long_name="Dew point temperature", **DEGC_TO_K,
        ),
        "wetblb_temp": cf.FieldSpec(
            var="wet_bulb_temperature", standard_name="wet_bulb_temperature",
            long_name="Wet bulb temperature", **DEGC_TO_K,
        ),
        "rel_hum": cf.FieldSpec(
            var="relative_humidity", standard_name="relative_humidity",
            long_name="Relative humidity",
        ),
        "stn_pres": cf.FieldSpec(
            var="air_pressure", standard_name="air_pressure",
            long_name="Station air pressure", **HPA_TO_PA,
        ),
        "mslp": cf.FieldSpec(
            var="air_pressure_at_mean_sea_level", standard_name="air_pressure_at_mean_sea_level",
            long_name="Mean sea level pressure", **HPA_TO_PA,
        ),
        "avg_wnd_spd_10m_pst10mts": cf.FieldSpec(
            var="wind_speed", standard_name="wind_speed",
            cell_methods="time: mean (interval: 10 minutes)",
            long_name="10 m wind speed, 10-minute mean", **KMH_TO_MS,
        ),
        "avg_wnd_dir_10m_pst10mts": cf.FieldSpec(
            var="wind_from_direction", standard_name="wind_from_direction",
            cell_methods="time: mean (interval: 10 minutes)",
            long_name="10 m wind direction, 10-minute mean",
        ),
        "max_wnd_spd_10m_pst10mts": cf.FieldSpec(
            var="wind_speed_of_gust", standard_name="wind_speed_of_gust",
            cell_methods="time: maximum (interval: 10 minutes)",
            long_name="10 m maximum wind gust, 10-minute window", **KMH_TO_MS,
        ),
        "pcpn_amt_pst1hr": cf.FieldSpec(
            var="precipitation_amount", standard_name="precipitation_amount",
            cell_methods="time: sum (interval: 1 hour)",
            long_name="Precipitation accumulated over the past hour", cf_units="kg m-2",
        ),
        "rnfl_amt_pst1hr": cf.FieldSpec(
            var="rainfall_amount", standard_name="thickness_of_rainfall_amount",
            cell_methods="time: sum (interval: 1 hour)",
            long_name="Rainfall accumulated over the past hour",
        ),
        "snw_dpth": cf.FieldSpec(
            var="surface_snow_thickness", standard_name="surface_snow_thickness",
            long_name="Snow depth", **CM_TO_M,
        ),
    }

    #: How far back to sample when enumerating stations.
    DISCOVERY_WINDOW = pd.Timedelta(hours=1)

    def discover_from_data(self, query: Query) -> list[StationMatch]:
        end = query.end
        start = max(query.start, end - self.DISCOVERY_WINDOW)
        params = {
            "bbox": ",".join(f"{v:.6f}" for v in (query.bbox or (-180, -90, 180, 90))),
            "datetime": (
                f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            ),
        }
        seen: dict[str, StationMatch] = {}
        for feature in paginate_ogc_items(
            self.items_url,
            params,
            provider=self.name,
            max_items=int(query.option("max_items", 50_000)),
        ):
            point = point_from_feature(feature)
            if point is None:
                continue
            lat, lon = point
            if not query.contains(lat, lon):
                continue
            props = feature.get("properties") or {}
            station_id = props.get("msc_id-value")
            if not station_id or str(station_id) in seen:
                continue
            match = self.new_match(
                station_id=str(station_id),
                name=str(props.get("stn_nam-value") or ""),
                lat=lat,
                lon=lon,
                variables=tuple(sorted(self.variables)),
                n_rows_est=int(query.days * self.samples_per_day),
                extra={"tc_id": props.get("tc_id-value")},
            )
            seen[str(station_id)] = match.attach_site(query)
        log.debug("eccc_swob discovered %d station(s)", len(seen))
        return list(seen.values())

    def is_qc_field(self, raw: str) -> bool:
        return raw.endswith(_SWOB_SUFFIXES)

    def qc_field_for(self, raw: str, spec: cf.FieldSpec) -> str | None:
        return f"{raw}-qa"

    def units_for(self, raw: str, rows: list[Mapping[str, Any]]) -> str | None:
        """Units come from the field's own ``-uom`` sibling, not from a table."""
        key = f"{raw}-uom"
        for row in rows:
            value = row.get(key)
            if value:
                return str(value)
        return None
