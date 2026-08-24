"""The discovery gate: filtering, per-site coverage, and the payload ceiling."""

from __future__ import annotations

import pytest

from omnisea.catalog import Catalog
from omnisea.errors import PayloadTooLargeError
from omnisea.providers.base import StationMatch
from omnisea.query import Query, Site

WEEK = ("2024-07-01", "2024-07-08")
BAMFIELD = Site(48.8353, -125.1358, "Bamfield")
VICTORIA = Site(48.42, -123.37, "Victoria")


def match(source="dfo_tides", station_id="08545", site="Bamfield", distance=0.5,
          rows=700, name="Bamfield", provider="dfo",
          variables=("water_surface_height_above_reference_datum",)):
    return StationMatch(
        source=source, provider=provider, station_id=station_id, name=name,
        lat=48.8353, lon=-125.1358, site=site, distance_km=distance,
        n_rows_est=rows, variables=variables,
    )


@pytest.fixture
def query():
    return Query.from_sites([BAMFIELD, VICTORIA], WEEK)


@pytest.fixture
def catalog(query):
    return Catalog(
        query,
        [
            match(station_id="08545", distance=0.1),
            match(station_id="08585", name="Effingham Bay", distance=13.8, rows=28),
            match(source="eccc_climate_daily", provider="eccc", station_id="1031316",
                  name="CAPE BEALE LIGHT", distance=8.0, rows=7,
                  variables=("air_temperature",)),
            match(station_id="07120", name="Victoria Harbour", site="Victoria", distance=0.5),
        ],
    )


class TestFiltering:
    def test_filter_by_source(self, catalog):
        assert len(catalog.filter(source="eccc_climate_daily")) == 1

    def test_filter_by_provider_selects_the_whole_organization(self, catalog):
        assert len(catalog.filter(provider="dfo")) == 3

    def test_filter_by_site(self, catalog):
        assert {m.station_id for m in catalog.filter(site="Victoria")} == {"07120"}

    def test_filter_by_variable(self, catalog):
        assert len(catalog.filter(variables="air_temperature")) == 1

    def test_filter_by_distance(self, catalog):
        assert len(catalog.filter(max_distance_km=1.0)) == 2

    def test_filter_by_name_substring(self, catalog):
        assert len(catalog.filter(name_contains="beale")) == 1

    def test_filtering_returns_a_new_catalog_and_leaves_the_original(self, catalog):
        before = len(catalog)
        catalog.filter(source="eccc_climate_daily")
        assert len(catalog) == before

    def test_nearest_is_applied_per_site_not_globally(self, catalog):
        """A global top-n would return several stations at one site and none at the other."""
        kept = catalog.filter(nearest=1)
        by_site = {m.site for m in kept}
        assert by_site == {"Bamfield", "Victoria"}
        assert len(kept) == 3  # one per (site, source) pair

    def test_custom_predicate(self, catalog):
        assert len(catalog.filter(where=lambda m: m.n_rows_est < 100)) == 2


class TestCoverage:
    def test_missing_sites_are_reported(self, query):
        catalog = Catalog(query, [match(site="Bamfield")])
        assert catalog.missing_sites == ["Victoria"]

    def test_no_missing_sites_when_all_matched(self, catalog):
        assert catalog.missing_sites == []

    def test_coverage_has_a_row_for_every_requested_site(self, query):
        catalog = Catalog(query, [match(site="Bamfield")])
        cov = catalog.coverage().set_index("site")
        assert set(cov.index) == {"Bamfield", "Victoria"}
        assert not bool(cov.loc["Victoria", "has_match"])

    def test_repr_names_the_sites_that_found_nothing(self, query):
        catalog = Catalog(query, [match(site="Bamfield")])
        assert "Victoria" in repr(catalog)

    def test_empty_catalog_repr_is_a_helpful_message(self, query):
        assert "no stations found" in repr(Catalog(query, []))


class TestPayloadCeiling:
    def test_oversized_catalogue_refuses_before_making_requests(self, query):
        catalog = Catalog(query, [match(rows=5_000_000)])
        with pytest.raises(PayloadTooLargeError) as excinfo:
            catalog.fetch(max_rows=1000)
        assert excinfo.value.estimate == 5_000_000

    def test_the_error_tells_you_which_knobs_to_turn(self, query):
        catalog = Catalog(query, [match(rows=5_000_000)])
        with pytest.raises(PayloadTooLargeError, match="nearest=1"):
            catalog.fetch(max_rows=1000)

    def test_row_estimate_is_the_sum_across_stations(self, catalog):
        assert catalog.n_rows_est == 700 + 28 + 7 + 700


class TestErrors:
    def test_discovery_errors_are_recorded_not_raised(self, query):
        catalog = Catalog(query, [match()], {"eccc_swob": "UpstreamError: HTTP 503"})
        assert "eccc_swob" in catalog.errors
        assert "503" in repr(catalog)

    def test_errors_survive_filtering(self, query):
        catalog = Catalog(query, [match()], {"eccc_swob": "boom"})
        assert catalog.filter(source="dfo_tides").errors == {"eccc_swob": "boom"}


class TestFrame:
    def test_frame_has_the_documented_columns(self, catalog):
        expected = {"source", "provider", "site", "station_id", "name", "lat", "lon",
                    "distance_km", "variables", "n_rows_est"}
        assert expected <= set(catalog.frame.columns)

    def test_frame_is_sorted_by_distance_for_a_site_query(self, catalog):
        distances = catalog.frame["distance_km"].tolist()
        assert distances == sorted(distances)

    def test_catalog_is_iterable_and_indexable(self, catalog):
        assert len(list(catalog)) == len(catalog)
        assert catalog[0].station_id
