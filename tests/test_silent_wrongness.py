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


def wide_frame(n=40, seed=7):
    """A model matrix with a planted near-duplicate cluster and one independent column."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-07-01", periods=n, freq="2h", name="time")
    base = np.sin(np.arange(n) / 5.0) * 4 + 15
    return pd.DataFrame({
        "temp_mean": base + rng.normal(0, 0.05, n),
        "temp_max": base + 3 + rng.normal(0, 0.05, n),
        "tide": np.sin(np.arange(n) / 1.9) + 2 + rng.normal(0, 0.3, n),
    }, index=idx)


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


# --------------------------------------------------------------------------- reporting


class TestCoverageDoesNotInventSites:
    """Found by a first-time user: one site in, two empty rows out.

    Labels are joined with ", " for netCDF and were then re-split on commas — so every
    auto-generated "lat,lon" label, and any name like "Tofino, BC", became several phantom
    sites all reported as having no data, in the one function whose whole job is showing gaps.
    """

    def test_an_auto_generated_lat_lon_label_stays_one_site(self):
        q = Query.from_position(49.153, -125.906, WEEK)
        rows = omnisea.coverage(build_tree(q, []))
        assert len(rows) == 1, f"one site requested, {len(rows)} reported"
        assert rows["site"].iloc[0] == "49.1530,-125.9060"

    def test_a_name_containing_a_comma_stays_one_site(self):
        q = Query.from_sites([Site(49.153, -125.906, "Tofino, BC")], WEEK)
        assert list(omnisea.coverage(build_tree(q, []))["site"]) == ["Tofino, BC"]

    def test_a_site_that_did_get_data_is_not_reported_empty(self):
        q = Query.from_sites([Site(48.8353, -125.1358, "Bamfield")], WEEK)
        idx = pd.date_range("2024-07-01", periods=4, freq="D", tz="UTC", name="time")
        tree = build_tree(q, [node("A", "in_situ/t", pd.DataFrame({"v": [1.0] * 4}, index=idx),
                                   {"v": {"units": "m"}})])
        rows = omnisea.coverage(tree)
        assert bool(rows["has_data"].iloc[0]), "a site with 4 timesteps reported as empty"

    def test_labels_survive_a_netcdf_round_trip(self, tmp_path):
        import xarray as xr

        pytest.importorskip("netCDF4")
        q = Query.from_position(49.153, -125.906, WEEK)
        path = tmp_path / "sites.nc"
        build_tree(q, []).to_netcdf(path)
        assert len(omnisea.coverage(xr.open_datatree(path))) == 1


class TestNothingCheckedIsNotACleanBillOfHealth:
    def test_correlations_says_when_min_overlap_excluded_everything(self, caplog):
        """8 grab samples in a week is an ordinary field sheet. An empty table read as
        'nothing is collinear' when the truth was 'nothing was examined'."""
        frame = wide_frame(n=8)
        with caplog.at_level("WARNING", logger="omnisea.align"):
            pairs = omnisea.correlations(frame)
        assert pairs.empty
        assert pairs.attrs["omnisea_pairs_below_min_overlap"] > 0
        assert any("nothing was checked" in r.message for r in caplog.records)

    def test_a_genuinely_uncorrelated_frame_warns_about_nothing(self, caplog):
        frame = wide_frame(n=40)[["tide"]].assign(other=np.arange(40.0) % 7)
        with caplog.at_level("WARNING", logger="omnisea.align"):
            omnisea.correlations(frame, threshold=0.99)
        assert not [r for r in caplog.records if "nothing was checked" in r.message]


class TestAnEmptyFetchExplainsItself:
    def test_a_query_matching_no_station_says_so(self):
        """discover()'s repr has always explained this; fetch() returned a bare empty tree."""
        q = Query.from_sites([Site(0.0, 0.0, "Null Island", radius_km=5)], WEEK)
        tree = Catalog(q, []).fetch()
        assert "No station matched" in tree.attrs["omnisea_empty_reason"]
        assert "No station matched" in omnisea.citation(tree)

    def test_a_failure_is_reported_instead_of_the_no_match_hint(self):
        q = Query.from_sites([Site(0.0, 0.0, "Null Island", radius_km=5)], WEEK)
        tree = Catalog(q, [], {"dfo_tides": "UpstreamError: HTTP 503"}).fetch()
        assert "omnisea_empty_reason" not in tree.attrs, "a real failure is not 'no match'"
        assert "503" in tree.attrs["omnisea_fetch_errors"]


class TestModelMatrix:
    def test_text_and_constant_columns_are_excluded_with_reasons(self):
        """align() is lossless, so it carries weather prose; sklearn dies on the first string."""
        frame = wide_frame(n=20)
        frame["WEATHER_ENG_DESC"] = "NA"
        frame["always_five"] = 5.0
        frame["all_missing"] = np.nan
        matrix = omnisea.model_matrix(frame)
        assert "WEATHER_ENG_DESC" not in matrix.columns
        assert "always_five" not in matrix.columns
        assert "all_missing" not in matrix.columns
        assert "tide" in matrix.columns
        excluded = matrix.attrs["omnisea_excluded"]
        assert "not numeric" in excluded["WEATHER_ENG_DESC"]
        assert "constant" in excluded["always_five"]

    def test_the_result_is_actually_usable_by_a_linear_model(self):
        frame = wide_frame(n=30)
        frame["WEATHER_ENG_DESC"] = "NA"
        matrix = omnisea.model_matrix(frame).dropna()
        # The advertised one-liner: straight into a least-squares fit, no cleaning.
        np.linalg.lstsq(matrix.to_numpy(dtype=float), np.arange(len(matrix), dtype=float),
                        rcond=None)

    def test_a_pinned_column_survives_whatever_it_looks_like(self):
        frame = wide_frame(n=20).assign(label="site-a")
        assert "label" in omnisea.model_matrix(frame, keep="label").columns


class TestColocatedStations:
    def test_nearest_prefers_the_longer_record_at_an_identical_position(self):
        """ECCC splits one physical site across station ids; picking arbitrarily cost a user
        46% of their temperature series with no signal."""
        from omnisea.catalog import _nearest_per_site

        short = StationMatch(source="eccc_climate", station_id="1038204", name="TOFINO A",
                             lat=49.08, lon=-125.77, site="T", distance_km=8.0, n_rows_est=92)
        long_ = StationMatch(source="eccc_climate", station_id="1038210", name="TOFINO A",
                             lat=49.08, lon=-125.77, site="T", distance_km=8.0, n_rows_est=169)
        assert [m.station_id for m in _nearest_per_site([short, long_], 1)] == ["1038210"]
        assert [m.station_id for m in _nearest_per_site([long_, short], 1)] == ["1038210"]

    def test_a_genuinely_closer_station_still_wins(self):
        from omnisea.catalog import _nearest_per_site

        near = StationMatch(source="s", station_id="near", name="A", lat=1, lon=1,
                            site="T", distance_km=1.0, n_rows_est=10)
        far = StationMatch(source="s", station_id="far", name="B", lat=2, lon=2,
                           site="T", distance_km=50.0, n_rows_est=10_000)
        assert [m.station_id for m in _nearest_per_site([far, near], 1)] == ["near"]


class TestMissingExtras:
    def test_writing_netcdf_without_the_engine_names_the_extra(self, monkeypatch, tmp_path):
        """The README promises "using a feature without its extra tells you exactly what to
        install"; the bare xarray error named two libraries the user never asked for."""
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        tree = build_tree(Query.from_sites([BAMFIELD], WEEK), [])
        with pytest.raises(omnisea.MissingDependencyError) as excinfo:
            omnisea.to_netcdf(tree, tmp_path / "x.nc")
        assert 'pip install "omnisea[netcdf]"' in str(excinfo.value)


class TestKeywordMistakes:
    def test_a_misspelled_position_keyword_names_the_keys_typed(self):
        """`latitude=`/`longitude=` used to die on "give one of bbox=, sites= or lat/lon="
        without ever mentioning the three keys the user actually typed."""
        with pytest.raises(QueryError) as excinfo:
            omnisea.discover(latitude=49.1, longitude=-125.9, time=WEEK)
        message = str(excinfo.value)
        assert "latitude" in message and "longitude" in message
        assert "did you mean" in message
