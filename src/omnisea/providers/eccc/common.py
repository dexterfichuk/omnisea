"""Unit conversions and shared field conventions for the ECCC collections."""

from __future__ import annotations

__all__ = [
    "DEGC_TO_K", "KMH_TO_MS", "KPA_TO_PA", "HPA_TO_PA", "MM_TO_KGM2", "CM_TO_M",
    "TENS_OF_DEGREES", "TENS_NOTE", "CLIMATE_SKIP",
]

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
    }
)

#: Backwards-compatible alias.
_CLIMATE_SKIP = CLIMATE_SKIP
