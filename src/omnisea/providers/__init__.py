"""Built-in providers and the classes third parties subclass to add their own."""

from __future__ import annotations

from .base import (
    DataSource,
    DiscoverySource,
    Provider,
    RetrievalSource,
    StationMatch,
    StationSeries,
)
from .cioos import CioosProvider
from .dfo import DfoProvider
from .eccc import EcccProvider
from .erddap import NAMED_PROVIDERS, ErddapProvider
from .noaa import CoopsProvider
from .ogc import OgcFeaturesProvider, OgcFeaturesSource
from .onc import OncProvider
from .usgs import UsgsProvider

__all__ = [
    "Provider",
    "DataSource",
    "RetrievalSource",
    "DiscoverySource",
    "StationMatch",
    "StationSeries",
    "OgcFeaturesProvider",
    "OgcFeaturesSource",
    "DfoProvider",
    "EcccProvider",
    "ErddapProvider",
    "CoopsProvider",
    "UsgsProvider",
    "NAMED_PROVIDERS",
    "CioosProvider",
    "OncProvider",
    "BUILTIN_PROVIDERS",
]

#: Instantiated at import; :func:`omnisea.registry.register_provider` wires them in.
#:
#: The ERDDAP installations omnisea knows are providers in their own right — ERDDAP is
#: software, and Hakai, IOOS and NOAA CoastWatch are organizations. That makes them ordinary:
#: they list in ``omnisea.sources()``, ``providers="hakai"`` selects one the way
#: ``providers="eccc"`` does, and an unqualified query sweeps the ones whose declared region
#: contains it. ``ErddapProvider`` itself stays for any installation not in that list, reached
#: with ``erddap_server=<url>``.
BUILTIN_PROVIDERS = [
    DfoProvider(), EcccProvider(), CioosProvider(), ErddapProvider(), OncProvider(),
    CoopsProvider(), UsgsProvider(),
    *(cls() for cls in NAMED_PROVIDERS.values()),
]
