"""CIOOS metadata records: both layouts, and the rules for rejecting a record."""

from __future__ import annotations

import json

import pytest
from conftest import FIXTURES

from omnisea.providers.cioos import CioosMetadataSource, CioosProvider, load_records, parse_record
from omnisea.query import Query

WEEK_2024 = ("2024-06-01", "2024-08-01")


@pytest.fixture
def source():
    return CioosMetadataSource(CioosProvider())


@pytest.fixture
def form_record():
    return json.loads((FIXTURES / "cioos_form_record.json").read_text())


class TestFormShape:
    def test_bilingual_title_takes_english(self, form_record):
        assert parse_record(form_record)["title"].startswith("Bamfield Inlet")

    def test_nsew_extent_becomes_a_bbox_and_centroid(self, form_record):
        record = parse_record(form_record)
        assert record["bbox"] == (-125.22, 48.78, -125.05, 48.90)
        assert 48.78 < record["lat"] < 48.90

    def test_distribution_urls_are_extracted(self, form_record):
        urls = parse_record(form_record)["distribution"]
        assert urls[0]["url"].endswith("bamfield_sst.html")

    def test_eovs_are_preserved(self, form_record):
        assert "seaSurfaceTemperature" in parse_record(form_record)["eov"]


class TestXmlShape:
    def test_metadata_xml_yaml_is_understood(self):
        pytest.importorskip("yaml")
        import yaml

        raw = yaml.safe_load((FIXTURES / "cioos_xml_record.yaml").read_text())
        record = parse_record(raw)
        assert record["title"] == "Barkley Sound mooring array"
        assert record["bbox"] == (-125.22, 48.78, -125.05, 48.90)
        assert "subSurfaceTemperature" in record["eov"]
        assert record["license"] == "CC-BY-4.0"
        assert record["organization"] == "Bamfield Marine Sciences Centre"


class TestRejection:
    def test_record_without_a_spatial_extent_is_skipped_not_guessed(self):
        raw = json.loads((FIXTURES / "cioos_no_extent.json").read_text())
        assert parse_record(raw) is None

    def test_polygon_only_extent_is_still_usable(self):
        raw = {
            "recordID": "poly-1",
            "title": {"en": "Polygon record"},
            "eov": ["oxygen"],
            "map": {"north": "", "south": "", "east": "", "west": "",
                    "polygon": "48.80,-125.20 48.90,-125.20 48.90,-125.05 48.80,-125.05"},
        }
        record = parse_record(raw)
        assert record is not None
        assert record["bbox"][0] == pytest.approx(-125.20)


class TestDiscovery:
    def test_unconfigured_source_is_silent_rather_than_failing(self, source):
        """There is no public CIOOS endpoint, so a plain discover() must not break."""
        query = Query.from_area((-126, 48, -125, 49), WEEK_2024)
        assert source.discover(query) == []

    def test_records_in_the_query_area_are_matched(self, source):
        query = Query.from_position(48.8353, -125.1358, WEEK_2024, radius_km=30,
                                    cioos_records=str(FIXTURES / "cioos_form_record.json"))
        matches = source.discover(query)
        assert [m.station_id for m in matches] == ["bamfield-sst-2024"]

    def test_records_outside_the_area_are_excluded(self, source):
        query = Query.from_position(48.42, -123.37, WEEK_2024, radius_km=10,
                                    cioos_records=str(FIXTURES / "cioos_form_record.json"))
        assert source.discover(query) == []

    def test_records_outside_the_time_window_are_excluded(self, source):
        query = Query.from_position(48.8353, -125.1358, ("2019-01-01", "2019-02-01"),
                                    radius_km=30,
                                    cioos_records=str(FIXTURES / "cioos_form_record.json"))
        assert source.discover(query) == []

    def test_eovs_are_translated_to_cf_names_on_the_match(self, source):
        query = Query.from_position(48.8353, -125.1358, WEEK_2024, radius_km=30,
                                    cioos_records=str(FIXTURES / "cioos_form_record.json"))
        (found,) = source.discover(query)
        assert "sea_surface_temperature" in found.variables

    def test_variable_filter_applies_through_the_eov_bridge(self, source):
        query = Query.from_position(48.8353, -125.1358, WEEK_2024, radius_km=30,
                                    variables=["sea_water_practical_salinity"],
                                    cioos_records=str(FIXTURES / "cioos_form_record.json"))
        assert source.discover(query) == []

    def test_a_directory_of_records_is_read(self, source):
        pytest.importorskip("yaml")
        query = Query.from_position(48.8353, -125.1358, WEEK_2024, radius_km=40,
                                    cioos_records=str(FIXTURES))
        assert len(source.discover(query)) >= 1

    def test_discovery_only_source_contributes_no_arrays(self, source):
        assert source.discovery_only is True
        assert source.fetch(None, []) == []


class TestLoading:
    def test_nested_firebase_style_export_is_flattened(self, form_record):
        """Firebase nests records region -> user -> id, so loading must walk the structure."""
        nested = {"region": {"user-1": {"rec-1": form_record}}}
        assert len(load_records(nested)) == 1

    def test_list_of_records_is_accepted(self, form_record):
        assert len(load_records([form_record, form_record])) == 2

    def test_unusable_spec_is_a_clear_error(self):
        from omnisea.errors import ProviderError

        with pytest.raises(ProviderError, match="could not interpret"):
            load_records("not a path, url or repo!!")
