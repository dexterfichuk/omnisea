"""The worked example from docs/adding-a-provider.md must actually work.

If this breaks, the documentation is lying to anyone trying to add a source.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import omnisea

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "csv_stations.py"


@pytest.fixture(scope="module")
def example_module():
    spec = importlib.util.spec_from_file_location("csv_stations_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "stations.csv").write_text(
        "id,name,lat,lon\n"
        "BAM01,Bamfield Inlet Logger,48.8353,-125.1358\n"
        "FAR01,Far Away Logger,50.0000,-128.0000\n",
        encoding="utf-8",
    )
    (tmp_path / "BAM01.csv").write_text(
        "time,water_temp_c,qc,battery_v\n"
        "2024-06-30T23:00:00Z,11.1,1,12.9\n"
        "2024-07-01T00:00:00Z,11.4,1,12.8\n"
        "2024-07-01T01:00:00Z,11.6,1,12.8\n"
        "2024-07-01T02:00:00Z,11.9,2,12.7\n"
        "2024-07-03T00:00:00Z,12.5,1,12.6\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def registered(example_module, data_dir):
    omnisea.register_provider(example_module.ShoreLoggerProvider(data_dir), replace=True)
    yield
    # Leave the registry as we found it so other tests are unaffected.
    from omnisea import registry

    registry._SOURCES.pop("shorelogger_sst", None)
    registry._PROVIDERS.pop("shorelogger", None)


BAMFIELD = dict(lat=48.8353, lon=-125.1358, radius_km=5)
DAY = ("2024-07-01", "2024-07-02")


def test_third_party_provider_appears_in_the_registry(registered):
    assert "shorelogger" in omnisea.providers()
    assert "shorelogger_sst" in omnisea.sources()


def test_it_is_discoverable_alongside_builtin_sources(registered):
    catalog = omnisea.discover(**BAMFIELD, time=DAY, providers="shorelogger")
    assert [m.station_id for m in catalog] == ["BAM01"]


def test_distant_station_is_excluded_by_radius(registered):
    catalog = omnisea.discover(**BAMFIELD, time=DAY, providers="shorelogger")
    assert "FAR01" not in {m.station_id for m in catalog}


def test_it_produces_a_cf_named_node(registered):
    tree = omnisea.fetch(**BAMFIELD, time=DAY, providers="shorelogger")
    ds = tree["/in_situ/shore_logger/BAM01"].dataset
    assert "sea_water_temperature" in ds.data_vars
    assert ds["sea_water_temperature"].attrs["standard_name"] == "sea_water_temperature"
    assert ds["sea_water_temperature"].attrs["units"] == "degC"


def test_unmapped_field_is_carried_through(registered):
    tree = omnisea.fetch(**BAMFIELD, time=DAY, providers="shorelogger")
    ds = tree["/in_situ/shore_logger/BAM01"].dataset
    assert "battery_v" in ds.data_vars
    assert ds["battery_v"].attrs["omnisea_mapped"] == 0


def test_qc_flags_are_carried(registered):
    tree = omnisea.fetch(**BAMFIELD, time=DAY, providers="shorelogger")
    assert "sea_water_temperature_qc" in tree["/in_situ/shore_logger/BAM01"].dataset.data_vars


def test_rows_outside_the_window_are_trimmed(registered):
    """The CSV holds rows before and after the requested day."""
    tree = omnisea.fetch(**BAMFIELD, time=DAY, providers="shorelogger")
    times = tree["/in_situ/shore_logger/BAM01"].dataset["time"].values
    assert len(times) == 3


def test_opt_in_unit_conversion_reaches_kelvin(registered):
    tree = omnisea.fetch(**BAMFIELD, time=DAY, providers="shorelogger", to_cf_units=True)
    ds = tree["/in_situ/shore_logger/BAM01"].dataset
    assert float(ds["sea_water_temperature"][0]) == pytest.approx(284.55)
    assert ds["sea_water_temperature"].attrs["units"] == "K"


def test_licence_from_the_provider_lands_on_the_node(registered):
    tree = omnisea.fetch(**BAMFIELD, time=DAY, providers="shorelogger")
    assert tree["/in_situ/shore_logger/BAM01"].attrs["license"] == "CC-BY-4.0"


def test_it_participates_in_multi_site_queries(registered):
    tree = omnisea.fetch(
        sites=[{"lat": 48.8353, "lon": -125.1358, "name": "Bamfield"},
               {"lat": 50.5, "lon": -128.5, "name": "Nowhere"}],
        radius_km=5, time=DAY, providers="shorelogger",
    )
    cov = omnisea.coverage(tree).set_index("site")
    assert bool(cov.loc["Bamfield", "has_data"])
    assert not bool(cov.loc["Nowhere", "has_data"])
