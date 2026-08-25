"""ECCC surface climate: hourly observations, and daily/monthly/seasonal/annual summaries.

Four of these collections are ordinary station time series and are wired here. Two related ones
are deliberately absent, for reasons worth recording next to the code:

* ``climate-normals`` is not a time series at all. It is one row per *(station, element, month)*
  carrying a single ``VALUE`` column named in prose by ``E_NORMAL_ELEMENT_NAME`` (66 elements ×
  13 months = 858 rows for one station), with ``MONTH`` 1–12 plus 13 for the annual figure and
  no time property of any kind. A ``datetime`` filter against it matches nothing, so the shared
  fetch path — which always sends one — would return an empty frame for every query. Turning it
  into a series would mean inventing a year to hang it on and pivoting 66 prose element names
  into columns. It needs its own source shape, not this one.
* ``ahccd-trends`` is one value per station, not per time — the same mismatch.
"""

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
    HPA_TO_PA,
    KMH_TO_MS,
    KPA_TO_PA,
    MM_TO_KGM2,
    TENS_NOTE,
    TENS_OF_DEGREES,
)

__all__ = [
    "EcccClimateHourly",
    "EcccClimateDaily",
    "EcccClimateMonthly",
    "EcccAhccdMonthly",
    "EcccAhccdSeasonal",
    "EcccAhccdAnnual",
]


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
    period = "D"
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


# --------------------------------------------------------------------------- monthly

#: The ``NORMAL_*`` columns are not observations. Each row repeats the station's 1981–2010
#: normal for that calendar month, so the same twelve numbers recur every year — they are the
#: reference the month is meant to be read against, not a second measurement of it.
NORMALS_NOTE = (
    "1981-2010 climate normal for this calendar month, repeated on every row of the station's "
    "record. Not an observation of this month."
)

#: A normal is a mean taken over the years of the normals period. CF spells that
#: ``time: <op> within years time: mean over years``; omnisea.align() reads the leading
#: operation, so a normal resamples the way its underlying quantity does.
NORMAL_MEAN = "time: mean within years time: mean over years"
NORMAL_SUM = "time: sum within years time: mean over years"


class EcccClimateMonthly(OgcFeaturesSource):
    """Monthly climate summaries — the multi-decadal view of the same stations.

    Two upstream facts shape this source, both confirmed against the live collection:

    ``LOCAL_DATE`` is ``"2022-08"``, a year and month with no day. Following
    :class:`EcccClimateDaily`, each monthly aggregate keeps its local calendar month and is
    stamped at 00:00Z on the *first* day of it; the convention is recorded in the node's
    ``time_reference`` attribute. A consequence worth knowing: the shared window trim keeps a
    month only when its first day falls inside the requested window, so a query starting
    mid-month begins at the following month rather than returning a month the window only
    partly covers.

    ``MLY_LAST_DATE`` in the station catalogue cannot be trusted. Station 1031316 is listed as
    ending 2007-02 while the collection serves it through 2026-07, and Ottawa CDA (6105976) is
    listed as having monthly data from 1889 while the collection holds none at all. So the end
    of the period of record is reported as open rather than used to reject stations — an
    understated end date would silently hide four decades of data.
    """

    name = "eccc_climate_monthly"
    title = "ECCC monthly climate summaries"
    node_path = "in_situ/weather_monthly"
    collection = "climate-monthly"
    station_collection = "climate-stations"
    station_id_field = "CLIMATE_IDENTIFIER"
    catalogue_id_field = "CLIMATE_IDENTIFIER"
    time_field = "LOCAL_DATE"
    skip_fields = CLIMATE_SKIP
    samples_per_day = 1.0 / 30.4375  # mean Gregorian month, so the estimate reads as months

    period = "M"

    fields = {
        # Verified against climate-daily for the same station-months: the monthly mean is the
        # mean of the daily means, and the min/max are the month's lowest and highest daily
        # extremes rather than means of them.
        "MEAN_TEMPERATURE": cf.FieldSpec(
            var="air_temperature", standard_name="air_temperature", units="degC",
            cell_methods="time: mean", long_name="Monthly mean air temperature", **DEGC_TO_K,
        ),
        "MIN_TEMPERATURE": cf.FieldSpec(
            var="air_temperature_min", standard_name="air_temperature", units="degC",
            cell_methods="time: minimum", long_name="Lowest daily minimum air temperature",
            **DEGC_TO_K,
        ),
        "MAX_TEMPERATURE": cf.FieldSpec(
            var="air_temperature_max", standard_name="air_temperature", units="degC",
            cell_methods="time: maximum", long_name="Highest daily maximum air temperature",
            **DEGC_TO_K,
        ),
        "NORMAL_MEAN_TEMPERATURE": cf.FieldSpec(
            var="air_temperature_normal", standard_name="air_temperature", units="degC",
            cell_methods=NORMAL_MEAN, long_name="Normal mean air temperature for this month",
            comment=NORMALS_NOTE, **DEGC_TO_K,
        ),
        "TOTAL_PRECIPITATION": cf.FieldSpec(
            var="precipitation_amount", standard_name="precipitation_amount", units="mm",
            cell_methods="time: sum", long_name="Monthly total precipitation", **MM_TO_KGM2,
        ),
        "NORMAL_PRECIPITATION": cf.FieldSpec(
            var="precipitation_amount_normal", standard_name="precipitation_amount", units="mm",
            cell_methods=NORMAL_SUM, long_name="Normal total precipitation for this month",
            comment=NORMALS_NOTE, **MM_TO_KGM2,
        ),
        "TOTAL_SNOWFALL": cf.FieldSpec(
            var="snowfall_amount", standard_name="thickness_of_snowfall_amount", units="cm",
            cell_methods="time: sum", long_name="Monthly total snowfall", **CM_TO_M,
        ),
        "NORMAL_SNOWFALL": cf.FieldSpec(
            var="snowfall_amount_normal", standard_name="thickness_of_snowfall_amount",
            units="cm", cell_methods=NORMAL_SUM,
            long_name="Normal total snowfall for this month", comment=NORMALS_NOTE, **CM_TO_M,
        ),
        "SNOW_ON_GROUND_LAST_DAY": cf.FieldSpec(
            var="surface_snow_thickness", standard_name="surface_snow_thickness", units="cm",
            long_name="Snow depth on the ground on the last day of the month",
            # Deliberately no cell_methods: this is a single reading, not a summary of the
            # month, so align() must treat it as a point value. It is the one variable here
            # whose observation time is the *end* of the interval its row is labelled with.
            comment=(
                "Observed on the last day of the month, while the row is stamped at the "
                "month's first day."
            ),
            **CM_TO_M,
        ),
        "BRIGHT_SUNSHINE": cf.FieldSpec(
            var="duration_of_sunshine", standard_name="duration_of_sunshine", units="h",
            cf_units="s", cf_scale=3600.0, cell_methods="time: sum",
            long_name="Total bright sunshine",
            comment="Discontinued at most stations; largely absent after the 1970s.",
        ),
        "NORMAL_SUNSHINE": cf.FieldSpec(
            var="duration_of_sunshine_normal", standard_name="duration_of_sunshine", units="h",
            cf_units="s", cf_scale=3600.0, cell_methods=NORMAL_SUM,
            long_name="Normal total bright sunshine for this month", comment=NORMALS_NOTE,
        ),
        "HEATING_DEGREE_DAYS": cf.FieldSpec(
            var="heating_degree_days",
            standard_name="integral_wrt_time_of_air_temperature_deficit",
            units="degC day", cell_methods="time: sum",
            long_name="Monthly total heating degree days",
        ),
        "COOLING_DEGREE_DAYS": cf.FieldSpec(
            var="cooling_degree_days",
            standard_name="integral_wrt_time_of_air_temperature_excess",
            units="degC day", cell_methods="time: sum",
            long_name="Monthly total cooling degree days",
        ),
        "DAYS_WITH_PRECIP_GE_1MM": cf.FieldSpec(
            var="days_with_precipitation_ge_1mm",
            standard_name=(
                "number_of_days_with_lwe_thickness_of_precipitation_amount_above_threshold"
            ),
            units="day", cell_methods="time: sum",
            long_name="Days with at least 1 mm of precipitation",
            comment="The threshold is 1 mm.",
        ),
        # The DAYS_WITH_VALID_* counts say how much of the month each aggregate was actually
        # built from. A monthly mean over 12 valid days is not comparable to one over 31, and
        # nothing else in the response records that, so they are named rather than left as
        # passthrough. No CF standard name exists for a coverage count.
        "DAYS_WITH_VALID_MEAN_TEMP": cf.FieldSpec(
            var="days_with_valid_mean_temperature", standard_name="", units="day",
            cell_methods="time: sum",
            long_name="Days contributing to the monthly mean temperature",
        ),
        "DAYS_WITH_VALID_MIN_TEMP": cf.FieldSpec(
            var="days_with_valid_min_temperature", standard_name="", units="day",
            cell_methods="time: sum",
            long_name="Days contributing to the monthly minimum temperature",
        ),
        "DAYS_WITH_VALID_MAX_TEMP": cf.FieldSpec(
            var="days_with_valid_max_temperature", standard_name="", units="day",
            cell_methods="time: sum",
            long_name="Days contributing to the monthly maximum temperature",
        ),
        "DAYS_WITH_VALID_PRECIP": cf.FieldSpec(
            var="days_with_valid_precipitation", standard_name="", units="day",
            cell_methods="time: sum",
            long_name="Days contributing to the monthly total precipitation",
        ),
        "DAYS_WITH_VALID_SNOWFALL": cf.FieldSpec(
            var="days_with_valid_snowfall", standard_name="", units="day",
            cell_methods="time: sum",
            long_name="Days contributing to the monthly total snowfall",
        ),
        "DAYS_WITH_VALID_SUNSHINE": cf.FieldSpec(
            var="days_with_valid_sunshine", standard_name="", units="day",
            cell_methods="time: sum",
            long_name="Days contributing to the monthly total bright sunshine",
        ),
    }

    def record_period(self, props: Mapping[str, Any]) -> tuple[Any, Any]:
        """Start only. See the class docstring for why ``MLY_LAST_DATE`` is not used."""
        return props.get("MLY_FIRST_DATE"), None

    def extract_time(self, row: Mapping[str, Any]) -> Any:
        """``"2022-08"`` becomes ``2022-08-01T00:00:00Z``. See the class docstring."""
        value = row.get(self.time_field)
        if not value:
            return None
        return f"{str(value)[:7]}-01T00:00:00Z"

    def datetime_param(self, query: Query) -> str:
        # The collection filters on local dates, so send plain dates rather than UTC instants.
        return f"{query.start.strftime('%Y-%m-%d')}/{query.end.strftime('%Y-%m-%d')}"

    def node_attrs(self, query: Query, match: StationMatch) -> dict[str, Any]:
        attrs = super().node_attrs(query, match)
        attrs["time_reference"] = (
            "LOCAL_DATE: monthly aggregates are labelled by local calendar month and stamped at "
            "00:00Z on the first day of that month. climate-monthly publishes no UTC date and "
            "no day of month, so no offset is applied. SNOW_ON_GROUND_LAST_DAY is the one "
            "exception: it is read on the last day of the month it is stamped at the start of."
        )
        return attrs


# --------------------------------------------------------------------------- AHCCD

#: AHCCD writes -9999.9 where a value is missing. Left alone it reads as a real measurement, so
#: it is nulled before shaping. Nothing genuine in these collections comes close: the coldest
#: temperature ever recorded in Canada is about -63 degC and pressures are hundreds of hPa.
AHCCD_MISSING = -9999.0

#: AHCCD publishes each measurement's units in a sibling property rather than in the collection
#: metadata. This maps value field to units field; it also supplies the skip list, so the units
#: fields cannot leak through as string variables of their own.
AHCCD_UNIT_FIELDS = {
    "temp_mean__temp_moyenne": "temp_mean_units__temp_moyenne_unites",
    "temp_max__temp_max": "temp_max_units__temp_max_unites",
    "temp_min__temp_min": "temp_min_units__temp_min_unites",
    "total_precip__precip_totale": "total_precip_units__precip_totale_unites",
    "rain__pluie": "rain_units__pluie_unites",
    "snow__neige": "snow_units__neige_unites",
    # Singular "unite" on this one. ECCC's own spelling, not a typo here.
    "pressure_sea_level__pression_niveau_mer": (
        "pressure_sea_level_units__pression_niveau_mer_unite"
    ),
    "pressure_station__pression_station": "pressure_station_units__pression_station_unites",
    "wind_speed__vitesse_vent": "wind_speed_units__vitesse_vent_unites",
}

AHCCD_SKIP = frozenset(
    {
        "identifier__identifiant",
        "station_id__id_station",
        "station_name__nom_station",
        "province__province",
        "period_group__groupe_periode",
        "period_value__valeur_periode",
        "date",
        "year__annee",
        # Position comes from the geometry, as everywhere else in this package.
        "lat__lat",
        "lon__long",
    }
) | frozenset(AHCCD_UNIT_FIELDS.values())

#: Seasons as AHCCD labels them, mapped to the calendar month each one *begins* in. Verified
#: against the monthly series: winter 1981 equals the mean of December 1980, January 1981 and
#: February 1981, so a winter is labelled with the year holding its January.
AHCCD_SEASON_START = {"Win": (-1, 12), "Spr": (0, 3), "Smr": (0, 6), "Fal": (0, 9)}


def _without_sentinels(row: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``row`` with AHCCD's -9999.9 missing marker replaced by ``None``."""
    return {
        key: (
            None
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value <= AHCCD_MISSING
            else value
        )
        for key, value in row.items()
    }


class _EcccAhccd(OgcFeaturesSource):
    """Shared shape of the AHCCD summary collections.

    AHCCD — Adjusted and Homogenized Canadian Climate Data — is the research-grade version of
    the same station network: the raw records corrected for station moves, instrument changes
    and known observing-practice shifts, so that a trend computed from them reflects climate
    rather than paperwork. The monthly, seasonal and annual collections publish identical
    properties and differ only in how a row is dated.

    Two things separate them from the ``climate-*`` collections:

    * Units travel per record, in a sibling property, rather than being fixed by the collection.
    * Missing values are written ``-9999.9`` instead of ``null``, which is why every row passes
      through :func:`_without_sentinels` before shaping.

    ``temp_max``/``temp_min`` are the month's **mean** daily maximum and minimum, not its
    extremes — the published ``temp_mean`` is their midpoint to within 0.2 degC across every row
    sampled. Their ``cell_methods`` therefore says ``time: mean``, and the "maximum within days"
    half of the CF compound spelling lives in ``long_name`` instead: omnisea.align() reads the
    leading operation, and a compound spelling would have it take a maximum over the months of
    a series of means.
    """

    #: AHCCD names its stations bilingually rather than in STATION_NAME.
    name_fields = ("station_name__nom_station", "STATION_NAME")
    station_collection = "ahccd-stations"
    station_id_field = "station_id__id_station"
    catalogue_id_field = "station_id__id_station"
    skip_fields = AHCCD_SKIP
    qc_suffix = ""  # AHCCD publishes no flag columns; the adjustment is the quality step

    fields = {
        "temp_mean__temp_moyenne": cf.FieldSpec(
            var="air_temperature", standard_name="air_temperature", units=None,
            cell_methods="time: mean", long_name="Mean air temperature (adjusted)",
            **DEGC_TO_K,
        ),
        "temp_max__temp_max": cf.FieldSpec(
            var="air_temperature_max", standard_name="air_temperature", units=None,
            cell_methods="time: mean", long_name="Mean daily maximum air temperature (adjusted)",
            **DEGC_TO_K,
        ),
        "temp_min__temp_min": cf.FieldSpec(
            var="air_temperature_min", standard_name="air_temperature", units=None,
            cell_methods="time: mean", long_name="Mean daily minimum air temperature (adjusted)",
            **DEGC_TO_K,
        ),
        "total_precip__precip_totale": cf.FieldSpec(
            var="precipitation_amount", standard_name="precipitation_amount", units=None,
            cell_methods="time: sum", long_name="Total precipitation (adjusted)", **MM_TO_KGM2,
        ),
        "rain__pluie": cf.FieldSpec(
            var="rainfall_amount", standard_name="thickness_of_rainfall_amount", units=None,
            cf_units="m", cf_scale=0.001, cell_methods="time: sum",
            long_name="Total rainfall (adjusted)",
        ),
        # Water equivalent, not snow depth: the published units are mm and rain + snow equals
        # total_precip exactly on every row sampled, which a depth in cm could not do. Hence
        # lwe_thickness_of_snowfall_amount rather than thickness_of_snowfall_amount.
        "snow__neige": cf.FieldSpec(
            var="snowfall_amount", standard_name="lwe_thickness_of_snowfall_amount", units=None,
            cf_units="m", cf_scale=0.001, cell_methods="time: sum",
            long_name="Total snowfall as water equivalent (adjusted)",
        ),
        "pressure_sea_level__pression_niveau_mer": cf.FieldSpec(
            var="air_pressure_at_sea_level", standard_name="air_pressure_at_mean_sea_level",
            units=None, cell_methods="time: mean",
            long_name="Mean sea-level air pressure (adjusted)", **HPA_TO_PA,
        ),
        "pressure_station__pression_station": cf.FieldSpec(
            var="air_pressure", standard_name="air_pressure", units=None,
            cell_methods="time: mean", long_name="Mean station air pressure (adjusted)",
            **HPA_TO_PA,
        ),
        "wind_speed__vitesse_vent": cf.FieldSpec(
            var="wind_speed", standard_name="wind_speed", units=None, cell_methods="time: mean",
            long_name="Mean wind speed (adjusted)", **KMH_TO_MS,
        ),
    }

    # ------------------------------------------------------------------ discovery

    def record_period(self, props: Mapping[str, Any]) -> tuple[Any, Any]:
        """Start only — ``end_date__date_fin`` describes the station's *trend* period.

        Every station sampled runs years past the end date the catalogue gives it: 1171020 is
        listed as ending 2004-03 and has monthly data through 2017-12. The start date matched
        the first record exactly in every case, so it is kept and the end is left open. This is
        the same trap ``MLY_LAST_DATE`` sets on :class:`EcccClimateMonthly`.
        """
        return props.get("start_date__date_debut"), None

    # ------------------------------------------------------------------ shaping

    def units_for(self, raw: str, rows: list[Mapping[str, Any]]) -> str | None:
        sibling = AHCCD_UNIT_FIELDS.get(raw)
        if not sibling:
            return None
        for row in rows:
            value = row.get(sibling)
            if value:
                return str(value)
        return None

    def clean_row(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        return _without_sentinels(row)

    def node_attrs(self, query: Query, match: StationMatch) -> dict[str, Any]:
        attrs = super().node_attrs(query, match)
        attrs["comment"] = (
            "Adjusted and Homogenized Canadian Climate Data: station records corrected for "
            "moves, instrument changes and observing-practice shifts so that trends reflect "
            "climate rather than station history. Values differ from the raw climate-* "
            "collections by design."
        )
        return attrs


class EcccAhccdMonthly(_EcccAhccd):
    """AHCCD monthly summaries — homogenized, and reaching back further than climate-monthly."""

    name = "eccc_ahccd_monthly"
    title = "ECCC AHCCD monthly summaries (adjusted)"
    node_path = "in_situ/ahccd_monthly"
    collection = "ahccd-monthly"
    time_field = "date"
    samples_per_day = 1.0 / 30.4375

    def extract_time(self, row: Mapping[str, Any]) -> Any:
        """``date`` is ``"1979-09"``; the month is stamped at 00:00Z on its first day."""
        value = row.get(self.time_field)
        if not value:
            return None
        return f"{str(value)[:7]}-01T00:00:00Z"

    def datetime_param(self, query: Query) -> str:
        return f"{query.start.strftime('%Y-%m-%d')}/{query.end.strftime('%Y-%m-%d')}"

    def node_attrs(self, query: Query, match: StationMatch) -> dict[str, Any]:
        attrs = super().node_attrs(query, match)
        attrs["time_reference"] = (
            "date: monthly aggregates are labelled by calendar month and stamped at 00:00Z on "
            "the first day of that month. AHCCD publishes no UTC date and no day of month."
        )
        return attrs


class EcccAhccdSeasonal(_EcccAhccd):
    """AHCCD seasonal summaries.

    A season is labelled by ``year__annee`` plus ``Win``/``Spr``/``Smr``/``Fal``, and winter is
    the awkward one: winter 1981 is December **1980** through February 1981, confirmed against
    the monthly series (its mean temperature equals the mean of those three months exactly).
    Each season is therefore stamped at 00:00Z on the first day of the month it begins in, which
    puts winter 1981 at 1980-12-01 — one interval start, uniformly, so a timestamp falling
    inside a season matches that season.

    That labelling disagrees with the upstream filter, which selects on the season's *year*, so
    the request is padded a year either side and the shared window trim decides what is kept.
    Without the pad, asking for a window ending in December would miss the winter that December
    starts.
    """

    name = "eccc_ahccd_seasonal"
    title = "ECCC AHCCD seasonal summaries (adjusted)"
    node_path = "in_situ/ahccd_seasonal"
    collection = "ahccd-seasonal"
    time_field = "year__annee"
    samples_per_day = 4.0 / 365.25
    datetime_pad = pd.DateOffset(years=1)

    def extract_time(self, row: Mapping[str, Any]) -> Any:
        year = row.get(self.time_field)
        season = AHCCD_SEASON_START.get(str(row.get("period_value__valeur_periode") or ""))
        if year in (None, "") or season is None:
            return None
        try:
            year = int(year)
        except (TypeError, ValueError):
            return None
        offset, month = season
        return f"{year + offset:04d}-{month:02d}-01T00:00:00Z"

    def datetime_param(self, query: Query) -> str:
        start = query.start - self.datetime_pad
        end = query.end + self.datetime_pad
        return f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

    def node_attrs(self, query: Query, match: StationMatch) -> dict[str, Any]:
        attrs = super().node_attrs(query, match)
        attrs["time_reference"] = (
            "year__annee + period_value__valeur_periode: each season is stamped at 00:00Z on "
            "the first day of the month it begins in (Win -> 1 December of the previous year, "
            "Spr -> 1 March, Smr -> 1 June, Fal -> 1 September). AHCCD labels winter with the "
            "year holding its January."
        )
        return attrs


class EcccAhccdAnnual(_EcccAhccd):
    """AHCCD annual summaries — one row per calendar year, stamped at 1 January."""

    name = "eccc_ahccd_annual"
    title = "ECCC AHCCD annual summaries (adjusted)"
    node_path = "in_situ/ahccd_annual"
    collection = "ahccd-annual"
    time_field = "year__annee"
    samples_per_day = 1.0 / 365.25

    def extract_time(self, row: Mapping[str, Any]) -> Any:
        value = row.get(self.time_field)
        if value in (None, ""):
            return None
        try:
            return f"{int(value):04d}-01-01T00:00:00Z"
        except (TypeError, ValueError):
            return None

    def datetime_param(self, query: Query) -> str:
        return f"{query.start.strftime('%Y-%m-%d')}/{query.end.strftime('%Y-%m-%d')}"

    def node_attrs(self, query: Query, match: StationMatch) -> dict[str, Any]:
        attrs = super().node_attrs(query, match)
        attrs["time_reference"] = (
            "year__annee: each calendar-year aggregate is stamped at 00:00Z on 1 January of "
            "that year. Verified against the monthly series: the annual mean is the mean of "
            "the twelve months of the same calendar year."
        )
        return attrs
