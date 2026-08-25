"""Regressions for defects found by an adversarial audit of the whole library.

Every test here pins a case where omnisea returned a **wrong or truncated answer without
saying so** — the one failure mode the library exists to prevent. They are collected in one
file rather than scattered because what they have in common is the failure mode, not the
module: a wrong number, a dropped station, a merged site, a truncated page.

Each test names the wrong behaviour it replaces, so a future reader can tell what the
assertion is defending and why it is worth the line.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

import omnisea
from omnisea import http
from omnisea.align import align
from omnisea.catalog import Catalog
from omnisea.errors import PayloadTooLargeError, QueryError
from omnisea.providers.base import StationMatch, StationSeries, frame_from_records
from omnisea.query import Query, Site
from omnisea.tree import build_tree

WEEK = ("2024-07-01", "2024-07-08")
BAMFIELD = Site(48.8353, -125.1358, "Bamfield", radius_km=30)
RAIN = {"precipitation_amount": {"units": "mm", "cell_methods": "time: sum"}}


def node(station_id, path, frame, var_attrs):
    match = StationMatch(source="t", provider="t", station_id=station_id, name=station_id,
                         lat=48.8353, lon=-125.1358, site="Bamfield")
    return StationSeries(match=match, frame=frame, node_path=f"{path}/{station_id}",
                         attrs={"provider": "t"}, var_attrs=var_attrs)


def utc(*stamps):
    return pd.DatetimeIndex(list(stamps), tz="UTC", name="time")


# --------------------------------------------------------------------------- wrong numbers


class TestASingleSampleIsNotStretched:
    """One 12 mm daily total used to become 12 mm on every following day — 72 mm a week."""

    def tree(self):
        q = Query.from_sites([BAMFIELD], WEEK)
        one = pd.DataFrame({"precipitation_amount": [12.0]}, index=utc("2024-07-01"))
        # A second node only to give the shared grid a span to cover.
        span = pd.DataFrame({"other": [0.0, 0.0]}, index=utc("2024-07-01", "2024-07-06"))
        return q, build_tree(q, [
            node("A", "in_situ/rain", one, RAIN),
            node("B", "in_situ/other", span, {"other": {"units": "1"}}),
        ])

    def test_the_total_is_not_multiplied_across_the_grid(self):
        _, tree = self.tree()
        column = align(tree, freq="1D")["precipitation_amount"]
        assert column.sum() == 12.0, f"12 mm became {column.sum()} mm"
        assert column.notna().sum() == 1, "the value belongs to its own day only"

    def test_the_audit_line_says_it_was_not_extended(self):
        _, tree = self.tree()
        out = align(tree, freq="1D")
        assert "not extended" in out.attrs["omnisea_aggregation"]["precipitation_amount@A"]


class TestASingleIntervalSampleRespectsTolerance:
    """A lone January daily total used to match every timestamp in July."""

    def tree(self):
        q = Query.from_sites([BAMFIELD], WEEK)
        january = pd.DataFrame({"precipitation_amount": [5.0]}, index=utc("2024-01-01"))
        return build_tree(q, [node("C", "in_situ/rain", january, RAIN)])

    def july(self, n=5):
        return pd.DataFrame({"time": pd.date_range("2024-07-01", periods=n, freq="D")})

    def test_a_six_month_old_reading_does_not_match_within_an_hour(self):
        got = align(self.tree(), on=self.july(), tolerance="1h")
        assert got["precipitation_amount"].isna().all(), "January matched July under 1h"

    def test_with_no_tolerance_the_unbounded_reach_is_declared(self):
        got = align(self.tree(), on=self.july(3))
        rule = got.attrs["omnisea_aggregation"]["precipitation_amount@C"]
        assert "UNBOUNDED" in rule, f"an unbounded match must say so; got {rule!r}"


class TestNumbersSpelledAsStringsStayNumbers:
    """One row's ``"8.2"`` made the column object dtype; it was then stored as *text*, and
    align() treated a daily mean as a category."""

    def test_a_mixed_numeric_column_is_coerced(self):
        frame = frame_from_records([
            {"time": "2024-07-01", "v": "8.1"},
            {"time": "2024-07-02", "v": 8.3},
        ])
        assert pd.api.types.is_numeric_dtype(frame["v"])
        assert frame["v"].tolist() == [8.1, 8.3]

    def test_prose_columns_are_left_alone(self):
        # Asserted on behaviour rather than dtype: pandas 3 infers StringDtype where 2.2 used
        # object, and what matters is that words did not become numbers.
        frame = frame_from_records([
            {"time": "2024-07-01", "weather": "CLOUDY"},
            {"time": "2024-07-02", "weather": "RAIN"},
        ])
        assert not pd.api.types.is_numeric_dtype(frame["weather"])
        assert frame["weather"].tolist() == ["CLOUDY", "RAIN"]

    def test_a_column_of_mostly_numbers_and_one_word_stays_prose(self):
        """Coercion must be all-or-nothing: a stray word means it was never a number column."""
        frame = frame_from_records([
            {"time": "2024-07-01", "v": "8.1"},
            {"time": "2024-07-02", "v": "TRACE"},
        ])
        assert not pd.api.types.is_numeric_dtype(frame["v"])
        assert frame["v"].tolist() == ["8.1", "TRACE"], "no value may be lost to coercion"


# --------------------------------------------------------------------------- dropped data


class TestPagingDoesNotStopAtAShortPage:
    """A server that caps `limit` made the FIRST page short — and 1500 of 2500 stations
    vanished with no error, no note, and a catalogue that looked complete."""

    def paged(self, sizes, matched, monkeypatch):
        pages, offset = {}, 0
        for size in sizes:
            pages[offset] = [{"id": offset + i} for i in range(size)]
            offset += size
        pages[offset] = []
        seen: list[int] = []

        def fake(url, params, provider=None, **kwargs):
            seen.append(params["offset"])
            return {"features": pages.get(params["offset"], []), "numberMatched": matched}

        monkeypatch.setattr(http, "get_json", fake)
        return seen

    def test_a_capped_first_page_does_not_end_the_collection(self, monkeypatch):
        self.paged([1000, 1000, 500], 2500, monkeypatch)
        got = list(http.paginate_ogc_items("u", {}, page_size=10_000))
        assert len(got) == 2500, f"silently truncated to {len(got)}"

    def test_a_short_page_mid_stream_does_not_end_it_either(self, monkeypatch):
        self.paged([1000, 700, 1000], 2700, monkeypatch)
        got = list(http.paginate_ogc_items("u", {}, page_size=1000))
        assert len(got) == 2700

    def test_numbermatched_still_ends_it_without_a_wasted_request(self, monkeypatch):
        seen = self.paged([1000, 1000], 2000, monkeypatch)
        assert len(list(http.paginate_ogc_items("u", {}, page_size=1000))) == 2000
        assert seen == [0, 1000], "a known total should not need a proving empty page"


class TestChunkTimeCannotHang:
    @pytest.mark.parametrize("max_days", [0, -1, 1e-18])
    def test_a_non_advancing_span_raises_instead_of_spinning(self, max_days):
        start = pd.Timestamp("2024-01-01", tz="UTC")
        with pytest.raises(ValueError, match="max_days"):
            http.chunk_time(start, start + pd.Timedelta(days=10), max_days=max_days)


class TestToDataframeSaysWhatItSkipped:
    def test_a_node_with_no_time_coordinate_is_reported(self, caplog):
        import xarray as xr

        q = Query.from_sites([BAMFIELD], WEEK)
        tree = build_tree(q, [])
        profile = xr.Dataset({"sea_water_temperature": ("depth", [9.1, 8.4, 7.2])},
                             coords={"depth": [0, 10, 20]})
        tree["/profiles/P1"] = xr.DataTree(profile)
        with caplog.at_level("WARNING", logger="omnisea.tree"):
            frame = omnisea.to_dataframe(tree)
        assert frame.empty
        assert any("no 'time' coordinate" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- merged sites


class TestSitesCannotSilentlyMerge:
    def test_two_sites_with_one_name_are_refused(self):
        """The label is the join key; merging two farms into one row loses a whole location."""
        with pytest.raises(QueryError, match="duplicate site label"):
            Query.from_sites([
                {"lat": 48.8, "lon": -125.1, "name": "Farm A"},
                {"lat": 50.0, "lon": -126.5, "name": "Farm A"},
            ], WEEK)

    def test_blank_names_in_a_csv_do_not_all_become_nan(self):
        csv = io.StringIO("name,lat,lon\n,48.8,-125.1\n,50.0,-126.5\n")
        sites = Query.from_sites(pd.read_csv(csv), WEEK).sites
        labels = [s.label for s in sites]
        assert labels == ["48.8000,-125.1000", "50.0000,-126.5000"], labels

    def test_a_blank_radius_column_is_refused_rather_than_matching_nothing(self):
        csv = io.StringIO("name,lat,lon,radius_km\nA,48.8,-125.1,10\nB,48.3,-123.5,\n")
        with pytest.raises(QueryError, match="radius_km is NaN"):
            Query.from_sites(pd.read_csv(csv), WEEK)


class TestPolarQueriesCoverEveryMeridian:
    def test_a_circle_spanning_all_longitudes_is_not_halved(self):
        """dlon was clamped to 180 and then applied as lon +/- 180, keeping half the meridians."""
        q = Query.from_position(89.9, 100.0, WEEK, radius_km=100)
        assert (q.bbox.west, q.bbox.east) == (-180.0, 180.0)
        assert q.contains(89.9, -100.0), "a station 22 km away across the pole"


class TestNaTWindows:
    def test_an_unparseable_end_date_is_refused(self):
        with pytest.raises(QueryError, match="NaT"):
            Query.from_area((-126, 48, -125, 49), ("2024-01-01", pd.NaT))


# --------------------------------------------------------------------------- readable output


class TestMessagesAndSchemas:
    def test_the_payload_error_is_prose_not_underscores(self):
        q = Query.from_sites([BAMFIELD], WEEK)
        match = StationMatch(source="dfo_tides", station_id="x", name="x",
                             lat=48.8, lon=-125.1, n_rows_est=5_000_000)
        with pytest.raises(PayloadTooLargeError) as excinfo:
            Catalog(q, [match]).fetch(max_rows=1000)
        message = str(excinfo.value)
        assert "station(s), over" in message, "prose commas became underscores"
        assert "max_rows=5_000_001" in message, "the number stays pasteable"

    def test_coverage_has_one_schema_whichever_way_you_asked(self):
        area = Catalog(Query.from_area((-126, 48, -125, 49), WEEK), []).coverage()
        site = Catalog(Query.from_sites([BAMFIELD], WEEK), []).coverage()
        assert list(area.columns) == list(site.columns)
        assert "has_match" in area.columns

    def test_summary_and_fields_are_addressable_when_empty(self):
        empty = build_tree(Query.from_sites([BAMFIELD], WEEK), [])
        assert omnisea.summary(empty)["node"].empty
        assert omnisea.fields(empty)["variable"].empty
        assert omnisea.stations(empty)["station_id"].empty


class TestCorrelationCountsFiniteValues:
    def test_n_is_what_the_correlation_was_actually_computed_on(self):
        """n is the number min_overlap is judged on — the defence against a spurious r=1.0."""
        n = 500
        frame = pd.DataFrame({
            "a": np.r_[np.arange(12.0), np.full(n - 12, np.inf)],
            "b": np.r_[np.arange(12.0) * 2, np.full(n - 12, np.inf)],
        })
        pairs = omnisea.correlations(frame, threshold=0.5)
        assert not pairs.empty
        assert pairs["n"].iloc[0] == 12, "n counted +/-inf rows the correlation ignored"


class TestAddLocalDoesNotOverwrite:
    def test_two_names_that_sanitize_alike_are_refused(self):
        """'Reef 1' and 'Reef.1' both become Reef_1; the first survey used to be destroyed."""
        tree = build_tree(Query.from_sites([BAMFIELD], WEEK), [])
        mine = pd.DataFrame({"time": pd.date_range("2024-07-01", periods=2), "x": [1.0, 2.0]})
        tree = omnisea.add_local(tree, mine, name="Reef 1", lat=48.8, lon=-125.1)
        with pytest.raises(QueryError, match="already has a node"):
            omnisea.add_local(tree, mine, name="Reef.1", lat=48.8, lon=-125.1)


class TestAlignRefusesDuplicateTimestamps:
    def test_the_error_names_the_node_and_the_instant(self):
        """It used to surface from inside pandas as "cannot reindex on an axis with
        duplicate labels", naming neither."""
        q = Query.from_sites([BAMFIELD], WEEK)
        frame = pd.DataFrame({"v": [1.0, 2.0]}, index=utc("2024-07-01", "2024-07-01"))
        tree = build_tree(q, [node("D", "in_situ/x", frame, {"v": {"units": "m"}})])
        with pytest.raises(QueryError, match="repeated timestamp"):
            align(tree, freq="1h")
