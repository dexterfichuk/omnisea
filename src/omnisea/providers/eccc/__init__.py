"""Environment and Climate Change Canada — the MSC GeoMet-OGC-API.

``https://api.weather.gc.ca`` (pygeoapi)

ECCC publishes over a hundred collections; omnisea wires the station-observation ones. They all
share the OGC API - Features plumbing in :mod:`omnisea.providers.ogc`, so each module here
declares only what is genuinely different about its datasets: the collection id, where the time
comes from, how a station is identified, and its field table.

Three upstream traps are handled, all confirmed by inspection:

1. ``climate-hourly`` unfiltered reports ``numberMatched`` of 276 million. Every request carries
   a station filter, and paging is capped with an explicit error rather than a silent truncation.
2. ``climate-stations`` publishes ``LATITUDE`` as integer micro-degrees (``483300000``), so
   coordinates are always read from ``geometry.coordinates``.
3. ``climate-daily`` has **no** ``UTC_DATE`` — only ``LOCAL_DATE`` — so its time convention is
   stated explicitly rather than guessed.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..ogc import OgcFeaturesProvider, OgcFeaturesSource
from .climate import EcccClimateDaily, EcccClimateHourly
from .hydrometric import (
    EcccHydrometric,
    EcccHydrometricAnnualPeaks,
    EcccHydrometricAnnualStatistics,
    EcccHydrometricDailyMean,
    EcccHydrometricMonthlyMean,
)
from .swob import EcccSwobRealtime

__all__ = [
    "EcccProvider",
    "EcccClimateHourly",
    "EcccClimateDaily",
    "EcccSwobRealtime",
    "EcccHydrometric",
    "EcccHydrometricDailyMean",
    "EcccHydrometricMonthlyMean",
    "EcccHydrometricAnnualStatistics",
    "EcccHydrometricAnnualPeaks",
]


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
            EcccHydrometricDailyMean(self),
            EcccHydrometricMonthlyMean(self),
            EcccHydrometricAnnualStatistics(self),
            EcccHydrometricAnnualPeaks(self),
        ]
