"""Shared fixtures. Everything here is a real, captured API response — trimmed, not synthesized."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# Bamfield Marine Sciences Centre, Barkley Sound, Vancouver Island — the running example.
BAMFIELD_LAT = 48.8353
BAMFIELD_LON = -125.1358
BAMFIELD_TIDE_STATION = "08545"


def load(name: str):
    path = FIXTURES / name
    if path.suffix in (".yaml", ".yml"):
        yaml = pytest.importorskip("yaml")
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def features(name: str) -> list[dict]:
    """The `properties` of each feature in a captured GeoJSON response."""
    return [f.get("properties", {}) for f in load(name).get("features", [])]


@pytest.fixture
def iwls_stations():
    return load("iwls_stations.json")


@pytest.fixture
def iwls_metadata():
    return load("iwls_metadata.json")


@pytest.fixture
def iwls_wlo():
    return load("iwls_wlo.json")


@pytest.fixture
def iwls_hilo():
    return load("iwls_hilo.json")


@pytest.fixture
def eccc_hourly_rows():
    return features("eccc_hourly.json")


@pytest.fixture
def eccc_daily_rows():
    return features("eccc_daily.json")


@pytest.fixture
def eccc_hydro_rows():
    return features("eccc_hydro.json")


@pytest.fixture
def eccc_swob_rows():
    return features("eccc_swob.json")


@pytest.fixture
def eccc_stations_geojson():
    return load("eccc_stations.json")
