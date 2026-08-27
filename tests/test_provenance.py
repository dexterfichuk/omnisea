"""Attribution derived from the result, so "what do I cite?" is answered by the data."""

from __future__ import annotations

import pandas as pd
import pytest

import omnisea
from omnisea.provenance import citation, provenance, sources_used
from omnisea.providers.base import StationMatch, StationSeries
from omnisea.query import Query, Site
from omnisea.tree import build_tree

WEEK = ("2024-07-01", "2024-07-08")
BAMFIELD = Site(48.8353, -125.1358, "Bamfield", radius_km=30)


def series(source, provider, station_id, institution, licence, terms, url):
    index = pd.date_range("2024-07-01", periods=4, freq="D", tz="UTC", name="time")
    match = StationMatch(source=source, provider=provider, station_id=station_id,
                         name=f"Station {station_id}", lat=48.8353, lon=-125.1358)
    return StationSeries(
        match=match,
        frame=pd.DataFrame({"water_level": [1.0, 2, 3, 4]}, index=index),
        node_path=f"in_situ/{source}/{station_id}",
        attrs={"provider": provider, "source_name": source, "institution": institution,
               "license": licence, "references": terms, "source_url": url},
    )


@pytest.fixture
def tree():
    q = Query.from_sites([BAMFIELD], WEEK)
    return build_tree(q, [
        series("dfo_tides", "dfo", "08545", "Fisheries and Oceans Canada",
               "OGL – Canada", "https://open.canada.ca/ogl", "https://iwls/stations/x/data"),
        series("eccc_climate_daily", "eccc", "1031316", "Environment and Climate Change Canada",
               "ECCC OGL", "https://eccc/licence", "https://api.weather.gc.ca/x"),
        series("eccc_climate_daily", "eccc", "1035940", "Environment and Climate Change Canada",
               "ECCC OGL", "https://eccc/licence", "https://api.weather.gc.ca/y"),
    ])


class TestProvenance:
    def test_one_row_per_source_by_default(self, tree):
        assert len(provenance(tree)) == 2

    def test_it_counts_the_stations_behind_each_source(self, tree):
        frame = provenance(tree).set_index("source")
        assert frame.loc["eccc_climate_daily", "n_stations"] == 2
        assert frame.loc["dfo_tides", "n_stations"] == 1

    def test_licence_and_terms_come_from_the_data(self, tree):
        frame = provenance(tree).set_index("source")
        assert frame.loc["dfo_tides", "license"] == "OGL – Canada"
        assert frame.loc["dfo_tides", "terms"] == "https://open.canada.ca/ogl"

    def test_grouping_by_provider_collapses_its_datasets(self, tree):
        frame = provenance(tree, by="provider").set_index("provider")
        assert frame.loc["eccc", "n_nodes"] == 2

    def test_node_detail_keeps_the_exact_url_each_series_came_from(self, tree):
        frame = provenance(tree, by="node")
        assert set(frame["source_url"]) == {
            "https://iwls/stations/x/data",
            "https://api.weather.gc.ca/x",
            "https://api.weather.gc.ca/y",
        }

    def test_time_span_is_reported(self, tree):
        frame = provenance(tree)
        assert frame["first"].min() == pd.Timestamp("2024-07-01")

    def test_an_unknown_grouping_is_rejected(self, tree):
        with pytest.raises(ValueError, match="by must be"):
            provenance(tree, by="sideways")

    def test_an_empty_tree_gives_an_empty_frame(self):
        assert provenance(build_tree(Query.from_sites([BAMFIELD], WEEK), [])).empty

    def test_sources_used_lists_the_datasets(self, tree):
        assert sources_used(tree) == ["dfo_tides", "eccc_climate_daily"]


@pytest.fixture
def erddap_like_tree():
    """One source holding datasets from two institutions under two licences.

    This is ERDDAP's normal condition — the licence belongs to the dataset, not the server —
    and the shape that makes "take the first institution per source" a mis-citation.
    """
    q = Query.from_sites([BAMFIELD], WEEK)
    return build_tree(q, [
        series("erddap_tabledap", "erddap", "ubc_mooring", "University of British Columbia",
               "CC-BY-4.0", "https://ubc/terms", "https://erddap/tabledap/ubc"),
        series("erddap_tabledap", "erddap", "hakai_buoy", "Hakai Institute",
               "CC-BY-SA-4.0", "https://hakai/terms", "https://erddap/tabledap/hakai"),
    ])


class TestMixedAttributionWithinOneSource:
    def test_distinct_institutions_are_not_collapsed(self, erddap_like_tree):
        frame = provenance(erddap_like_tree, by="source")
        assert len(frame) == 2
        assert set(frame["institution"]) == {
            "University of British Columbia", "Hakai Institute",
        }
        assert set(frame["license"]) == {"CC-BY-4.0", "CC-BY-SA-4.0"}

    def test_the_citation_names_both_licences(self, erddap_like_tree):
        text = citation(erddap_like_tree)
        assert "CC-BY-4.0" in text
        assert "CC-BY-SA-4.0" in text
        assert "Hakai Institute" in text

    def test_the_header_still_counts_one_source(self, erddap_like_tree):
        assert "from 1 source(s) across 2 station(s)" in citation(erddap_like_tree)

    def test_urls_are_listed_under_their_own_institution(self, erddap_like_tree):
        text = citation(erddap_like_tree, include_urls=True)
        # Each URL must appear exactly once — under its institution's entry, not under both.
        assert text.count("https://erddap/tabledap/ubc") == 1
        assert text.count("https://erddap/tabledap/hakai") == 1


class TestCitation:
    def test_it_names_every_institution(self, tree):
        text = citation(tree)
        assert "Fisheries and Oceans Canada" in text
        assert "Environment and Climate Change Canada" in text

    def test_it_states_the_licence_and_terms(self, tree):
        text = citation(tree)
        assert "OGL – Canada" in text
        assert "https://open.canada.ca/ogl" in text

    def test_it_records_the_retrieval_date(self, tree):
        """Realtime data cannot be recovered later from the query alone."""
        assert "omnisea" in citation(tree)
        assert str(pd.Timestamp.now(tz="UTC").year) in citation(tree)

    def test_it_reports_the_covered_window(self, tree):
        assert "2024-07-01 to 2024-07-04" in citation(tree)

    def test_urls_are_opt_in(self, tree):
        assert "https://iwls/stations/x/data" not in citation(tree)
        assert "https://iwls/stations/x/data" in citation(tree, include_urls=True)

    def test_markdown_style_is_a_list(self, tree):
        assert citation(tree, style="markdown").lstrip().startswith("**Data sources**")

    def test_an_unknown_style_is_rejected(self, tree):
        with pytest.raises(ValueError, match="style must be"):
            citation(tree, style="bibtex")

    def test_an_empty_tree_says_so_rather_than_producing_a_stub(self):
        assert "empty" in citation(build_tree(Query.from_sites([BAMFIELD], WEEK), []))

    def test_an_incomplete_retrieval_is_flagged_in_the_citation(self, tree):
        """You should not quietly publish a partial pull."""
        tree.attrs["omnisea_fetch_errors"] = "eccc_swob: UpstreamError HTTP 503"
        assert "incomplete" in citation(tree).lower()

    def test_it_is_exported(self):
        assert omnisea.citation is citation
        assert omnisea.provenance is provenance
