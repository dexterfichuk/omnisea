"""Environment and Climate Change Canada — the MSC GeoMet-OGC-API.

``https://api.weather.gc.ca`` (pygeoapi)

One provider, four datasets: hourly climate, daily climate, SWOB realtime and hydrometric
realtime. They share the OGC API - Features plumbing in :mod:`omnisea.providers.ogc`; what
differs is declared per class below.

Three upstream traps are handled here, all confirmed by inspection:

1. ``climate-hourly`` unfiltered reports ``numberMatched`` of 276 million. Every request carries
   a station filter, and paging is capped with an explicit error rather than a silent truncation.
2. ``climate-stations`` publishes ``LATITUDE`` as integer micro-degrees (``483300000``), so
   coordinates are always read from ``geometry.coordinates``.
3. ``climate-daily`` has **no** ``UTC_DATE`` — only ``LOCAL_DATE`` — so its time convention is
   stated explicitly rather than guessed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .. import cf
from ..http import paginate_ogc_items
from ..query import Query
from .base import StationMatch
from .ogc import OgcFeaturesProvider, OgcFeaturesSource, point_from_feature

log = logging.getLogger("omnisea.eccc")

__all__ = ["EcccProvider"]


class EcccProvider(OgcFeaturesProvider):
    name = "eccc"
    title = "Environment and Climate Change Canada / Meteorological Service of Canada"
    base_url = "https://api.weather.gc.ca"
    license = "Environment and Climate Change Canada — Open Government Licence – Canada"
    terms_url = "https://eccc-msc.github.io/open-data/licence/readme_en/"

    def build_sources(self) -> Sequence[OgcFeaturesSource]:
        return [
            EcccClimateHourly(self),
            EcccClimateDaily(self),
            EcccSwobRealtime(self),
            EcccHydrometric(self),
        ]


# --------------------------------------------------------------------------- climate hourly


DEGC_TO_K = dict(cf_units="K", cf_offset=273.15)
KMH_TO_MS = dict(cf_units="m s-1", cf_scale=1.0 / 3.6)
KPA_TO_PA = dict(cf_units="Pa", cf_scale=1000.0)
HPA_TO_PA = dict(cf_units="Pa", cf_scale=100.0)
MM_TO_KGM2 = dict(cf_units="kg m-2", cf_scale=1.0)
CM_TO_M = dict(cf_units="m", cf_scale=0.01)
TENS_OF_DEGREES = dict(scale=10.0, units="degree", cf_units="degree")

_TENS_NOTE = "ECCC publishes this in tens of degrees; omnisea multiplies by 10."

_CLIMATE_SKIP = frozenset(
    {
        "STATION_NAME",
        "CLIMATE_IDENTIFIER",
        "ID",
        "PROVINCE_CODE",
        "STN_ID",
        "LOCAL_DATE",
        "LOCAL_YEAR",
        "LOCAL_MONTH",
        "LOCAL_DAY",
        "LOCAL_HOUR",
        "UTC_DATE",
        "UTC_YEAR",
        "UTC_MONTH",
        "UTC_DAY",
        "LATITUDE_DECIMAL_DEGREES",
        "LONGITUDE_DECIMAL_DEGREES",
    }
)


class EcccClimateHourly(OgcFeaturesSource):
    """Hourly surface climate observations from the adjusted station network."""

    name = "eccc_climate"
    title = "ECCC hourly climate observations"
    node_path = "in_situ/weather"
    collection = "climate-hourly"
    station_collection = "climate-stations"
    station_id_field = "CLIMATE_IDENTIFIER"
    catalogue_id_field = "CLIMATE_IDENTIFIER"
    time_field = "UTC_DATE"
    skip_fields = _CLIMATE_SKIP
    samples_per_day = 24.0
    require_record_period = True  # HLY_FIRST_DATE is null for stations with no hourly record
    #: climate-hourly filters on LOCAL_DATE even though it publishes UTC_DATE, so a UTC window
    #: comes back shifted by the station's offset. Ask for a day either side and trim to the
    #: real window afterwards; one extra day is 24 rows.
    datetime_pad = pd.Timedelta(days=1)

    fields = {
        "TEMP": cf.FieldSpec(
            var="air_temperature", standard_name="air_temperature", units="degC",
            long_name="Air temperature", qc_field="TEMP_FLAG", **DEGC_TO_K,
        ),
        "DEW_POINT_TEMP": cf.FieldSpec(
            var="dew_point_temperature", standard_name="dew_point_temperature", units="degC",
            long_name="Dew point temperature", qc_field="DEW_POINT_TEMP_FLAG", **DEGC_TO_K,
        ),
        "RELATIVE_HUMIDITY": cf.FieldSpec(
            var="relative_humidity", standard_name="relative_humidity", units="percent",
            long_name="Relative humidity", qc_field="RELATIVE_HUMIDITY_FLAG",
        ),
        "STATION_PRESSURE": cf.FieldSpec(
            var="air_pressure", standard_name="air_pressure", units="kPa",
            long_name="Station air pressure", qc_field="STATION_PRESSURE_FLAG", **KPA_TO_PA,
        ),
        "WIND_SPEED": cf.FieldSpec(
            var="wind_speed", standard_name="wind_speed", units="km h-1",
            long_name="Wind speed", qc_field="WIND_SPEED_FLAG", **KMH_TO_MS,
        ),
        "WIND_DIRECTION": cf.FieldSpec(
            var="wind_from_direction", standard_name="wind_from_direction",
            long_name="Wind direction (from)", qc_field="WIND_DIRECTION_FLAG",
            comment=_TENS_NOTE, **TENS_OF_DEGREES,
        ),
        "PRECIP_AMOUNT": cf.FieldSpec(
            var="precipitation_amount", standard_name="precipitation_amount", units="mm",
            long_name="Precipitation amount", cell_methods="time: sum",
            qc_field="PRECIP_AMOUNT_FLAG", **MM_TO_KGM2,
        ),
        "VISIBILITY": cf.FieldSpec(
            var="visibility_in_air", standard_name="visibility_in_air", units="km",
            cf_units="m", cf_scale=1000.0, long_name="Horizontal visibility",
            qc_field="VISIBILITY_FLAG",
        ),
        "WINDCHILL": cf.FieldSpec(
            var="wind_chill_temperature", standard_name="wind_chill_of_air_temperature",
            units="degC", long_name="Wind chill", qc_field="WINDCHILL_FLAG", **DEGC_TO_K,
        ),
        "HUMIDEX": cf.FieldSpec(
            var="humidex", standard_name="", units="degC",  # no CF standard name for humidex
            long_name="Humidex", qc_field="HUMIDEX_FLAG", **DEGC_TO_K,
        ),
    }

    def record_period(self, props: Mapping[str, Any]) -> tuple[Any, Any]:
        return props.get("HLY_FIRST_DATE"), props.get("HLY_LAST_DATE")

    def datetime_param(self, query: Query) -> str:
        start = query.start - self.datetime_pad
        end = query.end + self.datetime_pad
        return (
            f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )


class EcccClimateDaily(OgcFeaturesSource):
    """Daily climate summaries.

    ``climate-daily`` publishes **no** ``UTC_DATE``. Rather than invent an offset, each daily
    aggregate keeps its local calendar date and is stamped at ``00:00Z``; the convention is
    recorded in the node's ``time_reference`` attribute. Converting a daily statistic to a UTC
    instant would imply a precision the data does not have.
    """

    name = "eccc_climate_daily"
    title = "ECCC daily climate summaries"
    node_path = "in_situ/weather_daily"
    collection = "climate-daily"
    station_collection = "climate-stations"
    station_id_field = "CLIMATE_IDENTIFIER"
    catalogue_id_field = "CLIMATE_IDENTIFIER"
    time_field = "LOCAL_DATE"
    skip_fields = _CLIMATE_SKIP
    samples_per_day = 1.0
    require_record_period = True  # DLY_FIRST_DATE is null for stations with no daily record

    fields = {
        "MEAN_TEMPERATURE": cf.FieldSpec(
            var="air_temperature", standard_name="air_temperature", units="degC",
            cell_methods="time: mean", long_name="Daily mean air temperature",
            qc_field="MEAN_TEMPERATURE_FLAG", **DEGC_TO_K,
        ),
        "MIN_TEMPERATURE": cf.FieldSpec(
            var="air_temperature_min", standard_name="air_temperature", units="degC",
            cell_methods="time: minimum", long_name="Daily minimum air temperature",
            qc_field="MIN_TEMPERATURE_FLAG", **DEGC_TO_K,
        ),
        "MAX_TEMPERATURE": cf.FieldSpec(
            var="air_temperature_max", standard_name="air_temperature", units="degC",
            cell_methods="time: maximum", long_name="Daily maximum air temperature",
            qc_field="MAX_TEMPERATURE_FLAG", **DEGC_TO_K,
        ),
        "TOTAL_PRECIPITATION": cf.FieldSpec(
            var="precipitation_amount", standard_name="precipitation_amount", units="mm",
            cell_methods="time: sum", long_name="Daily total precipitation",
            qc_field="TOTAL_PRECIPITATION_FLAG", **MM_TO_KGM2,
        ),
        "TOTAL_RAIN": cf.FieldSpec(
            var="rainfall_amount", standard_name="thickness_of_rainfall_amount", units="mm",
            cf_units="m", cf_scale=0.001, cell_methods="time: sum",
            long_name="Daily total rainfall", qc_field="TOTAL_RAIN_FLAG",
        ),
        "TOTAL_SNOW": cf.FieldSpec(
            var="snowfall_amount", standard_name="thickness_of_snowfall_amount", units="cm",
            cell_methods="time: sum", long_name="Daily total snowfall",
            qc_field="TOTAL_SNOW_FLAG", **CM_TO_M,
        ),
        "SNOW_ON_GROUND": cf.FieldSpec(
            var="surface_snow_thickness", standard_name="surface_snow_thickness", units="cm",
            long_name="Snow depth on the ground", qc_field="SNOW_ON_GROUND_FLAG", **CM_TO_M,
        ),
        "SPEED_MAX_GUST": cf.FieldSpec(
            var="wind_speed_of_gust", standard_name="wind_speed_of_gust", units="km h-1",
            cell_methods="time: maximum", long_name="Daily maximum wind gust speed",
            qc_field="SPEED_MAX_GUST_FLAG", **KMH_TO_MS,
        ),
        "DIRECTION_MAX_GUST": cf.FieldSpec(
            var="wind_from_direction_of_gust", standard_name="wind_from_direction",
            long_name="Direction of the daily maximum wind gust",
            qc_field="DIRECTION_MAX_GUST_FLAG", comment=_TENS_NOTE, **TENS_OF_DEGREES,
        ),
        "MIN_REL_HUMIDITY": cf.FieldSpec(
            var="relative_humidity_min", standard_name="relative_humidity", units="percent",
            cell_methods="time: minimum", long_name="Daily minimum relative humidity",
            qc_field="MIN_REL_HUMIDITY_FLAG",
        ),
        "HEATING_DEGREE_DAYS": cf.FieldSpec(
            var="heating_degree_days",
            standard_name="integral_wrt_time_of_air_temperature_deficit",
            units="degC day", long_name="Heating degree days",
            qc_field="HEATING_DEGREE_DAYS_FLAG",
        ),
        "COOLING_DEGREE_DAYS": cf.FieldSpec(
            var="cooling_degree_days",
            standard_name="integral_wrt_time_of_air_temperature_excess",
            units="degC day", long_name="Cooling degree days",
            qc_field="COOLING_DEGREE_DAYS_FLAG",
        ),
    }

    def record_period(self, props: Mapping[str, Any]) -> tuple[Any, Any]:
        return props.get("DLY_FIRST_DATE"), props.get("DLY_LAST_DATE")

    def extract_time(self, row: Mapping[str, Any]) -> Any:
        """Keep the local calendar date, stamped at midnight UTC. See the class docstring."""
        value = row.get(self.time_field)
        if not value:
            return None
        return f"{str(value)[:10]}T00:00:00Z"

    def datetime_param(self, query: Query) -> str:
        # The collection filters on local dates, so send plain dates rather than UTC instants.
        return f"{query.start.strftime('%Y-%m-%d')}/{query.end.strftime('%Y-%m-%d')}"

    def node_attrs(self, query: Query, match: StationMatch) -> dict[str, Any]:
        attrs = super().node_attrs(query, match)
        attrs["time_reference"] = (
            "LOCAL_DATE: daily aggregates are labelled by local calendar date and stamped at "
            "00:00Z. climate-daily publishes no UTC_DATE, so no offset is applied."
        )
        return attrs


# --------------------------------------------------------------------------- SWOB realtime


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


# --------------------------------------------------------------------------- hydrometric


class EcccHydrometric(OgcFeaturesSource):
    """Realtime water level and discharge from the national hydrometric network."""

    name = "eccc_hydrometric"
    title = "ECCC hydrometric realtime"
    node_path = "in_situ/hydrometric"
    collection = "hydrometric-realtime"
    station_collection = "hydrometric-stations"
    station_id_field = "STATION_NUMBER"
    catalogue_id_field = "STATION_NUMBER"
    time_field = "DATETIME"
    skip_fields = frozenset(
        {
            "IDENTIFIER",
            "STATION_NUMBER",
            "STATION_NAME",
            "PROV_TERR_STATE_LOC",
            "DATETIME",
            "DATETIME_LST",
        }
    )
    qc_suffix = ""
    samples_per_day = 24.0 * 4  # 15-minute reporting is typical

    fields = {
        "LEVEL": cf.FieldSpec(
            var="water_surface_height_above_reference_datum",
            standard_name="water_surface_height_above_reference_datum",
            units="m", long_name="Water level", qc_field="LEVEL_SYMBOL_EN",
        ),
        "DISCHARGE": cf.FieldSpec(
            var="water_volume_transport_in_river_channel",
            standard_name="water_volume_transport_in_river_channel",
            units="m3 s-1", long_name="River discharge", qc_field="DISCHARGE_SYMBOL_EN",
        ),
    }

    def is_qc_field(self, raw: str) -> bool:
        return raw.endswith(("_SYMBOL_EN", "_SYMBOL_FR"))

    def qc_field_for(self, raw: str, spec: cf.FieldSpec) -> str | None:
        return spec.qc_field

    def station_from_feature(
        self, query: Query, feature: Mapping[str, Any]
    ) -> StationMatch | None:
        match = super().station_from_feature(query, feature)
        if match is None:
            return None
        props = feature.get("properties") or {}
        # The catalogue lists discontinued gauges too; realtime data only exists for active ones.
        if props.get("REAL_TIME") in (0, "0", False):
            return None
        match.extra["vertical_datum"] = props.get("VERTICAL_DATUM") or None
        match.extra["status"] = props.get("STATUS_EN")
        return match

    def node_attrs(self, query: Query, match: StationMatch) -> dict[str, Any]:
        attrs = super().node_attrs(query, match)
        datum = match.extra.get("vertical_datum")
        if datum:
            attrs["datum"] = datum
        return attrs
