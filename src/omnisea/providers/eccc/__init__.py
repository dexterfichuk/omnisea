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
from datetime import timedelta

from ...http import NEVER_CACHE
from ..ogc import OgcFeaturesProvider, OgcFeaturesSource
from .climate import (
    EcccAhccdAnnual,
    EcccAhccdMonthly,
    EcccAhccdSeasonal,
    EcccClimateDaily,
    EcccClimateHourly,
    EcccClimateMonthly,
)
from .hydrometric import (
    EcccHydrometric,
    EcccHydrometricAnnualPeaks,
    EcccHydrometricAnnualStatistics,
    EcccHydrometricDailyMean,
    EcccHydrometricMonthlyMean,
)
from .swob import EcccSwobMarine, EcccSwobRealtime

__all__ = [
    "EcccProvider",
    "EcccClimateHourly",
    "EcccClimateDaily",
    "EcccClimateMonthly",
    "EcccAhccdMonthly",
    "EcccAhccdSeasonal",
    "EcccAhccdAnnual",
    "EcccSwobRealtime",
    "EcccSwobMarine",
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

    #: Order matters — first match wins. The realtime collections are excluded outright (a
    #: stale observation is a wrong number, not a slow one) and are listed before the broad
    #: globs so nothing can claim them first. Station catalogues are near-static; the appended
    #: archives are quality-controlled and published days behind real time, so an hour of
    #: staleness cannot hide a measurement that was available when the query ran; the annual
    #: products are re-issued about once a year. ``ahccd-stations`` is deliberately caught by
    #: the ``*-stations`` rule above the ``ahccd-*`` one.
    cache_policy = {
        "*/collections/swob-realtime/items": NEVER_CACHE,
        "*/collections/hydrometric-realtime/items": NEVER_CACHE,
        "*/collections/*-stations/items": timedelta(days=7),
        "*/collections/climate-hourly/items": timedelta(hours=1),
        "*/collections/climate-daily/items": timedelta(hours=1),
        "*/collections/climate-monthly/items": timedelta(hours=1),
        "*/collections/hydrometric-daily-mean/items": timedelta(hours=1),
        "*/collections/hydrometric-monthly-mean/items": timedelta(hours=1),
        "*/collections/ahccd-*/items": timedelta(days=1),
        "*/collections/hydrometric-annual-*/items": timedelta(days=1),
    }

    def build_sources(self) -> Sequence[OgcFeaturesSource]:
        return [
            EcccClimateHourly(self),
            EcccClimateDaily(self),
            EcccClimateMonthly(self),
            EcccAhccdMonthly(self),
            EcccAhccdSeasonal(self),
            EcccAhccdAnnual(self),
            EcccSwobRealtime(self),
            EcccSwobMarine(self),
            EcccHydrometric(self),
            EcccHydrometricDailyMean(self),
            EcccHydrometricMonthlyMean(self),
            EcccHydrometricAnnualStatistics(self),
            EcccHydrometricAnnualPeaks(self),
        ]
