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
from .erddap import ErddapProvider
from .ogc import OgcFeaturesProvider, OgcFeaturesSource
from .onc import OncProvider

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
    "CioosProvider",
    "OncProvider",
    "BUILTIN_PROVIDERS",
]

#: Instantiated at import; :func:`omnisea.registry.register_provider` wires them in.
BUILTIN_PROVIDERS = [
    DfoProvider(), EcccProvider(), CioosProvider(), ErddapProvider(), OncProvider(),
]
