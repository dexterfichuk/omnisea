"""ECCC surface climate: hourly observations and daily summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ... import cf
from ...query import Query
from ..base import StationMatch
from ..ogc import OgcFeaturesSource
from .common import (
    CLIMATE_SKIP,
    CM_TO_M,
    DEGC_TO_K,
    KMH_TO_MS,
    KPA_TO_PA,
    MM_TO_KGM2,
    TENS_NOTE,
    TENS_OF_DEGREES,
)

__all__ = ["EcccClimateHourly", "EcccClimateDaily"]


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
    skip_fields = CLIMATE_SKIP
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
            comment=TENS_NOTE, **TENS_OF_DEGREES,
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
    skip_fields = CLIMATE_SKIP
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
            qc_field="DIRECTION_MAX_GUST_FLAG", comment=TENS_NOTE, **TENS_OF_DEGREES,
        ),
        "MIN_REL_HUMIDITY": cf.FieldSpec(
            var="relative_humidity_min", standard_name="relative_humidity", units="percent",
            cell_methods="time: minimum", long_name="Daily minimum relative humidity",
            qc_field="MIN_REL_HUMIDITY_FLAG",
        ),
        "HEATING_DEGREE_DAYS": cf.FieldSpec(
            var="heating_degree_days",
            standard_name="integral_wrt_time_of_air_temperature_deficit",
            units="degC day", cell_methods="time: sum", long_name="Heating degree days",
            qc_field="HEATING_DEGREE_DAYS_FLAG",
        ),
        "COOLING_DEGREE_DAYS": cf.FieldSpec(
            var="cooling_degree_days",
            standard_name="integral_wrt_time_of_air_temperature_excess",
            units="degC day", cell_methods="time: sum", long_name="Cooling degree days",
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
