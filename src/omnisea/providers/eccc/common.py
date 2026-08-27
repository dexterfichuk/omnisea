"""Unit conversions and shared field conventions for the ECCC collections."""

from __future__ import annotations

__all__ = [
    "DEGC_TO_K", "KMH_TO_MS", "KPA_TO_PA", "HPA_TO_PA", "MM_TO_KGM2", "CM_TO_M",
    "TENS_OF_DEGREES", "TENS_NOTE", "CLIMATE_SKIP", "PROVINCE_TIME_ZONES", "time_zone_for",
]

from collections.abc import Mapping
from typing import Any

DEGC_TO_K = dict(cf_units="K", cf_offset=273.15)
KMH_TO_MS = dict(cf_units="m s-1", cf_scale=1.0 / 3.6)
KPA_TO_PA = dict(cf_units="Pa", cf_scale=1000.0)
HPA_TO_PA = dict(cf_units="Pa", cf_scale=100.0)
MM_TO_KGM2 = dict(cf_units="kg m-2", cf_scale=1.0)
CM_TO_M = dict(cf_units="m", cf_scale=0.01)

#: ECCC stores wind direction in tens of degrees; 25 means 250 degrees.
TENS_OF_DEGREES = dict(scale=10.0, units="degree", cf_units="degree")
TENS_NOTE = "ECCC publishes this in tens of degrees; omnisea multiplies by 10."

CLIMATE_SKIP = frozenset(
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
        # climate-monthly publishes these as decimal-degree *strings*; unskipped they become
        # variables competing with the geometry the coordinates actually come from.
        "LATITUDE",
        "LONGITUDE",
        "LAST_UPDATED",
        "ENG_PROVINCE_NAME",
        "FRE_PROVINCE_NAME",
    }
)


#: Which IANA timezone a station's *local* calendar date is expressed in.
#:
#: ECCC's ``climate-daily`` and ``climate-monthly`` label every row by local calendar date and
#: publish no UTC date at all, so reading one requires knowing the station's timezone. The
#: station metadata gives a province code but no zone, and the two are not quite the same
#: question — a few corners of the country keep a different clock from the rest of their
#: province. This is exact for the coastal stations omnisea is built around and approximate at
#: those margins; :func:`time_zone_for` narrows the two worst by longitude.
#:
#: Checked against the data rather than assumed: for 1,369 station-days at 15 stations during
#: daylight time, binning ECCC's hourly precipitation by *civil* local date reproduced the
#: published daily total on 1,273 of them, against 982 for Local Standard Time.
PROVINCE_TIME_ZONES = {
    "NL": "America/St_Johns",
    "NS": "America/Halifax",
    "PE": "America/Halifax",
    "NB": "America/Moncton",
    "QC": "America/Toronto",
    "ON": "America/Toronto",
    "MB": "America/Winnipeg",
    "SK": "America/Regina",      # Saskatchewan does not observe DST.
    "AB": "America/Edmonton",
    "BC": "America/Vancouver",
    "YT": "America/Whitehorse",  # UTC-7 year-round since 2020.
    "NT": "America/Yellowknife",
    "NU": "America/Iqaluit",     # Nunavut spans three zones; this is the eastern one.
}


def time_zone_for(props: Mapping[str, Any], lon: float | None) -> str:
    """The IANA zone a station's local dates are in, or ``""`` if it cannot be determined.

    Falls back to the caller's own estimate (``align()`` uses longitude) when the province is
    missing or unrecognised, which is the case for a station outside Canada.
    """
    code = str(props.get("PROV_STATE_TERR_CODE") or props.get("PROVINCE_CODE") or "").upper()
    zone = PROVINCE_TIME_ZONES.get(code, "")
    if not zone or lon is None:
        return zone
    # The two provinces wide enough for the province code alone to put a station in the wrong
    # zone by a whole hour. Northwestern Ontario keeps Central time; southeastern British
    # Columbia keeps Mountain.
    if code == "ON" and lon < -90.0:
        return "America/Winnipeg"
    if code == "BC" and lon > -120.0:
        return "America/Edmonton"
    return zone
