"""Time alignment: turning a ragged tree into a model-ready rectangle.

The point of these tests is that resampling is chosen from CF metadata rather than guessed —
an accumulation must never be interpolated, and an extreme must never be averaged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import omnisea
from omnisea.align import add_local, aggregation_for, align
from omnisea.errors import QueryError
from omnisea.providers.base import StationMatch, StationSeries
from omnisea.query import Query, Site
from omnisea.tree import build_tree

WEEK = ("2024-07-01", "2024-07-08")
BAMFIELD = Site(48.8353, -125.1358, "Bamfield", radius_km=30)


def series(station_id, node, frame, var_attrs=None, name="Test"):
    match = StationMatch(
        source="test", provider="test", station_id=station_id, name=name,
        lat=48.8353, lon=-125.1358, site="Bamfield",
    )
    return StationSeries(
        match=match, frame=frame, node_path=f"{node}/{station_id}",
        attrs={"Conventions": "CF-1.10", "provider": "test"},
        var_attrs=var_attrs or {},
    )


@pytest.fixture
def ragged_tree():
    """Three cadences, as a real query produces: 15-minute, daily, and irregular."""
    q = Query.from_sites([BAMFIELD], WEEK)

    fine_idx = pd.date_range("2024-07-01", "2024-07-08", freq="15min", tz="UTC", name="time")
    fine = pd.DataFrame({"water_level": np.sin(np.arange(len(fine_idx)) / 6.0) + 2.0},
                        index=fine_idx)

    daily_idx = pd.date_range("2024-07-01", "2024-07-07", freq="D", tz="UTC", name="time")
    daily = pd.DataFrame(
        {"precipitation_amount": [0.0, 5.0, 0.0, 12.0, 0.0, 0.0, 3.0],
         "air_temperature": [14.0, 14.5, 13.8, 15.2, 16.0, 14.8, 17.1],
         "air_temperature_max": [16.5, 17.0, 16.5, 19.0, 20.0, 25.5, 22.0]},
        index=daily_idx,
    )

    return build_tree(q, [
        series("08545", "in_situ/tides", fine,
               {"water_level": {"units": "m"}}),  # no cell_methods -> instantaneous
        series("1031316", "in_situ/weather_daily", daily, {
            "precipitation_amount": {"units": "mm", "cell_methods": "time: sum"},
            "air_temperature": {"units": "degC", "cell_methods": "time: mean"},
            "air_temperature_max": {"units": "degC", "cell_methods": "time: maximum"},
        }),
    ])


class TestAggregationRules:
    def test_accumulation_sums_and_never_interpolates(self):
        assert aggregation_for({"cell_methods": "time: sum"}) == ("sum", "ffill")

    def test_maximum_takes_the_max_not_the_mean(self):
        """The max of daily maxima is a real maximum; their mean is a statistic of nothing."""
        assert aggregation_for({"cell_methods": "time: maximum"})[0] == "max"

    def test_minimum_takes_the_min(self):
        assert aggregation_for({"cell_methods": "time: minimum"})[0] == "min"

    def test_interval_mean_forward_fills_rather_than_interpolating(self):
        """A daily mean spread across its own day is honest; interpolating invents structure."""
        assert aggregation_for({"cell_methods": "time: mean"}) == ("mean", "ffill")

    def test_instantaneous_value_may_be_interpolated(self):
        assert aggregation_for({}) == ("mean", "interpolate")

    def test_explicit_point_is_instantaneous(self):
        assert aggregation_for({"cell_methods": "time: point"})[1] == "interpolate"

    def test_non_numeric_takes_first_and_ffills(self):
        assert aggregation_for({}, numeric=False) == ("first", "ffill")


class TestResampleToGrid:
    def test_downsampling_averages_an_instantaneous_series(self, ragged_tree):
        hourly = align(ragged_tree, freq="1h")
        assert hourly.attrs["omnisea_aggregation"]["water_level@08545"] == "mean"

    def test_downsampling_sums_an_accumulation(self, ragged_tree):
        """Weekly precipitation must be the sum of the daily totals, not their mean."""
        weekly = align(ragged_tree, freq="7D")
        total = weekly["precipitation_amount"].sum()
        assert total == pytest.approx(20.0)

    def test_downsampling_maxima_takes_the_largest(self, ragged_tree):
        weekly = align(ragged_tree, freq="7D")
        assert weekly["air_temperature_max"].max() == pytest.approx(25.5)

    def test_upsampling_an_accumulation_forward_fills(self, ragged_tree):
        """Interpolating a daily total to hourly would invent an intra-day distribution."""
        hourly = align(ragged_tree, freq="1h")
        rule = hourly.attrs["omnisea_aggregation"]["precipitation_amount@1031316"]
        assert "ffill" in rule
        july2 = hourly.loc["2024-07-02", "precipitation_amount"].dropna()
        assert (july2 == 5.0).all()

    def test_upsampling_an_instantaneous_series_interpolates(self, ragged_tree):
        fine = align(ragged_tree, freq="5min")
        assert "interpolate" in fine.attrs["omnisea_aggregation"]["water_level@08545"]

    def test_result_is_one_row_per_grid_step(self, ragged_tree):
        hourly = align(ragged_tree, freq="1h")
        assert isinstance(hourly.index, pd.DatetimeIndex)
        assert hourly.index.freqstr in ("h", "H")

    def test_every_column_records_how_it_got_there(self, ragged_tree):
        hourly = align(ragged_tree, freq="1h")
        applied = hourly.attrs["omnisea_aggregation"]
        assert applied, "resampling choices must be auditable"
        assert len(applied) == hourly.shape[1]


class TestJoinToOwnTimestamps:
    @pytest.fixture
    def field_sheet(self):
        return pd.DataFrame({
            "time": pd.to_datetime(["2024-07-01 09:14", "2024-07-02 10:05",
                                    "2024-07-04 14:20", "2024-07-06 16:30"]),
            "chlorophyll_ug_L": [5.6, 1.8, 6.8, 5.4],
        })

    def test_your_own_columns_are_carried_through(self, ragged_tree, field_sheet):
        """You want y and X in one table, not two you have to line up yourself."""
        joined = align(ragged_tree, on=field_sheet)
        assert "chlorophyll_ug_L" in joined.columns
        assert list(joined["chlorophyll_ug_L"]) == [5.6, 1.8, 6.8, 5.4]

    def test_one_row_per_supplied_timestamp(self, ragged_tree, field_sheet):
        assert len(align(ragged_tree, on=field_sheet)) == len(field_sheet)

    def test_daily_values_match_by_interval_containment(self, ragged_tree, field_sheet):
        """A sample at 10:05 belongs to that day's total, not to a value 10 hours away."""
        joined = align(ragged_tree, on=field_sheet, tolerance="30min")
        assert joined["precipitation_amount"].notna().all()
        assert joined["precipitation_amount"].iloc[1] == pytest.approx(5.0)  # July 2

    def test_a_short_tolerance_does_not_starve_the_daily_columns(self, ragged_tree, field_sheet):
        """The bug this guards: 1h tolerance vs a midnight stamp returned all-NaN."""
        joined = align(ragged_tree, on=field_sheet, tolerance="1h")
        assert joined["air_temperature"].notna().all()

    def test_instantaneous_values_respect_the_tolerance(self, ragged_tree, field_sheet):
        joined = align(ragged_tree, on=field_sheet, tolerance="30min")
        assert "within 30min" in joined.attrs["omnisea_aggregation"]["water_level@08545"]

    def test_match_counts_are_reported(self, ragged_tree, field_sheet):
        joined = align(ragged_tree, on=field_sheet)
        assert "4/4 matched" in joined.attrs["omnisea_aggregation"]["water_level@08545"]

    def test_a_datetime_index_works_as_the_target(self, ragged_tree):
        target = pd.date_range("2024-07-01", periods=5, freq="D")
        assert len(align(ragged_tree, on=target)) == 5

    def test_a_time_column_is_found_without_being_named(self, ragged_tree, field_sheet):
        assert len(align(ragged_tree, on=field_sheet)) == 4

    def test_tz_aware_input_is_converted_not_rejected(self, ragged_tree, field_sheet):
        aware = field_sheet.copy()
        aware["time"] = aware["time"].dt.tz_localize("America/Vancouver")
        joined = align(ragged_tree, on=aware)
        assert joined["water_level"].notna().any()

    def test_frame_without_usable_times_is_a_clear_error(self, ragged_tree):
        with pytest.raises(QueryError, match="DatetimeIndex or a time column"):
            align(ragged_tree, on=pd.DataFrame({"value": [1, 2, 3]}))

    def test_duplicate_timestamps_are_rejected(self, ragged_tree):
        dupes = pd.to_datetime(["2024-07-01", "2024-07-01"])
        with pytest.raises(QueryError, match="unique"):
            align(ragged_tree, on=dupes)


class TestColumnNaming:
    def test_unambiguous_variables_keep_their_bare_name(self, ragged_tree):
        assert "water_level" in align(ragged_tree, freq="1h").columns

    def test_qualified_always_names_the_station(self, ragged_tree):
        cols = align(ragged_tree, freq="1h", columns="qualified").columns
        assert "water_level@08545" in cols

    def test_multi_gives_a_variable_station_index(self, ragged_tree):
        wide = align(ragged_tree, freq="1h", columns="multi")
        assert wide.columns.names == ["variable", "station"]

    def test_same_variable_at_two_stations_is_disambiguated(self):
        q = Query.from_sites([BAMFIELD], WEEK)
        idx = pd.date_range("2024-07-01", periods=24, freq="h", tz="UTC", name="time")
        tree = build_tree(q, [
            series("A", "in_situ/weather", pd.DataFrame({"air_temperature": range(24)}, index=idx)),
            series("B", "in_situ/weather", pd.DataFrame({"air_temperature": range(24)}, index=idx)),
        ])
        cols = align(tree, freq="1h").columns
        assert "air_temperature@A" in cols and "air_temperature@B" in cols

    def test_one_station_in_two_branches_does_not_collide(self):
        """Observed tides and predicted extrema are both station 08545."""
        q = Query.from_sites([BAMFIELD], WEEK)
        idx = pd.date_range("2024-07-01", periods=24, freq="h", tz="UTC", name="time")
        tree = build_tree(q, [
            series("08545", "in_situ/tides", pd.DataFrame({"reviewed": [1] * 24}, index=idx)),
            series("08545", "predictions/tides_hilo",
                   pd.DataFrame({"reviewed": [1] * 24}, index=idx)),
        ])
        wide = align(tree, freq="1h")
        assert not wide.columns.duplicated().any()

    def test_unknown_style_is_rejected(self, ragged_tree):
        with pytest.raises(QueryError, match="columns must be"):
            align(ragged_tree, freq="1h", columns="sideways")


class TestArgumentChecking:
    def test_freq_and_on_are_mutually_exclusive(self, ragged_tree):
        with pytest.raises(QueryError, match="exactly one"):
            align(ragged_tree, freq="1h", on=pd.date_range("2024-07-01", periods=3))

    def test_one_of_them_is_required(self, ragged_tree):
        with pytest.raises(QueryError, match="exactly one"):
            align(ragged_tree)

    def test_qc_columns_are_excluded_by_default(self):
        q = Query.from_sites([BAMFIELD], WEEK)
        idx = pd.date_range("2024-07-01", periods=24, freq="h", tz="UTC", name="time")
        tree = build_tree(q, [series("A", "in_situ/tides", pd.DataFrame(
            {"water_level": range(24), "water_level_qc": ["1"] * 24}, index=idx))])
        assert "water_level_qc" not in align(tree, freq="1h").columns
        assert any("water_level_qc" in c for c in align(tree, freq="1h", include_qc=True).columns)

    def test_explicit_override_wins_over_cell_methods(self, ragged_tree):
        weekly = align(ragged_tree, freq="7D", agg={"precipitation_amount": "max"})
        assert weekly["precipitation_amount"].iloc[0] == pytest.approx(12.0)

    def test_empty_tree_gives_an_empty_frame(self):
        q = Query.from_sites([BAMFIELD], WEEK)
        assert align(build_tree(q, []), freq="1h").empty


class TestAddLocal:
    @pytest.fixture
    def samples(self):
        return pd.DataFrame({
            "time": pd.to_datetime(["2024-07-01 09:14", "2024-07-03 10:05"]),
            "chlorophyll_ug_L": [5.6, 1.8],
        })

    def test_your_data_becomes_a_node(self, ragged_tree, samples):
        merged = add_local(ragged_tree, samples, name="Grab samples",
                           lat=48.8353, lon=-125.1358, station_id="BAM-CHL")
        assert "/in_situ/local/BAM-CHL" in merged.groups

    def test_existing_nodes_survive(self, ragged_tree, samples):
        merged = add_local(ragged_tree, samples, name="Grab samples",
                           lat=48.8353, lon=-125.1358, station_id="BAM-CHL")
        assert "/in_situ/tides/08545" in merged.groups

    def test_your_attributes_are_kept(self, ragged_tree, samples):
        merged = add_local(ragged_tree, samples, name="Grab samples", lat=48.8353,
                           lon=-125.1358, station_id="BAM-CHL",
                           var_attrs={"chlorophyll_ug_L": {"units": "ug L-1"}})
        ds = merged["/in_situ/local/BAM-CHL"].dataset
        assert ds["chlorophyll_ug_L"].attrs["units"] == "ug L-1"

    def test_your_cell_methods_drive_alignment_like_any_provider(self, ragged_tree, samples):
        merged = add_local(ragged_tree, samples, name="Daily totals", lat=48.8353,
                           lon=-125.1358, station_id="MINE",
                           var_attrs={"chlorophyll_ug_L": {"cell_methods": "time: sum"}})
        weekly = align(merged, freq="7D")
        assert weekly["chlorophyll_ug_L"].iloc[0] == pytest.approx(7.4)

    def test_it_round_trips_to_netcdf(self, ragged_tree, samples, tmp_path):
        pytest.importorskip("netCDF4")
        merged = add_local(ragged_tree, samples, name="Grab samples",
                           lat=48.8353, lon=-125.1358, station_id="BAM-CHL")
        path = tmp_path / "local.nc"
        merged.to_netcdf(path)
        assert "/in_situ/local/BAM-CHL" in xr.open_datatree(path).groups

    def test_frame_without_times_is_a_clear_error(self, ragged_tree):
        with pytest.raises(QueryError, match="DatetimeIndex or a time column"):
            add_local(ragged_tree, pd.DataFrame({"v": [1]}), name="x", lat=0.0, lon=0.0)


def test_align_is_exported():
    assert omnisea.align is align
    assert omnisea.add_local is add_local


class TestRegressions:
    """Each of these is a bug that shipped and silently corrupted a column."""

    def _tree(self, frame, node="in_situ/a", sid="X", attrs=None):
        frame = frame.copy()
        frame.index.name = "time"
        return build_tree(
            Query.from_sites([BAMFIELD], WEEK), [series(sid, node, frame, attrs)]
        )

    def test_upsampling_an_irregular_series_keeps_its_values(self):
        """Binning an irregular series discarded every value: none sit on a grid boundary.

        Tidal extrema are *always* at irregular times (03:33, 09:58, ...), so the whole
        column came back NaN while every other column looked fine.
        """
        idx = pd.DatetimeIndex(
            ["2024-07-01 03:33", "2024-07-01 09:58", "2024-07-01 16:12", "2024-07-01 22:40"]
        )
        tree = self._tree(pd.DataFrame({"extremum": [3.1, 0.4, 2.8, 0.9]}, index=idx))
        hourly = align(tree, freq="1h")
        assert hourly["extremum"].notna().sum() > 0, "irregular values were discarded"
        # Interpolation samples the curve *at* grid points, so a peak occurring at 03:33 is
        # approached but never reproduced exactly. What matters is that the shape survives.
        assert 2.5 < hourly["extremum"].max() <= 3.1
        assert 0.4 <= hourly["extremum"].min() < 1.0

    def test_real_tidal_extrema_survive_resampling(self):
        """The shape that actually broke: ~6 h irregular spacing upsampled to hourly."""
        idx = pd.DatetimeIndex(
            ["2024-07-01 03:33", "2024-07-01 09:58", "2024-07-01 16:12",
             "2024-07-02 01:40", "2024-07-02 08:15"]
        )
        tree = self._tree(pd.DataFrame({"extremum": [3.1, 0.4, 2.8, 0.9, 3.3]}, index=idx))
        assert align(tree, freq="1h")["extremum"].notna().sum() >= 20

    def test_single_point_series_does_not_crash_the_grid(self):
        """min == max produced a duplicate-labelled span index and a reindex ValueError."""
        idx = pd.DatetimeIndex(["2024-07-03 12:00"])
        tree = self._tree(pd.DataFrame({"v": [42.0]}, index=idx))
        assert align(tree, freq="1h")["v"].notna().sum() == 1

    def test_unbounded_join_is_labelled_as_such(self):
        """A lone reading otherwise becomes a constant column across the whole query."""
        idx = pd.DatetimeIndex(["2024-07-03 12:00"])
        tree = self._tree(pd.DataFrame({"v": [42.0]}, index=idx))
        joined = align(tree, on=pd.date_range("2024-07-01", periods=5, freq="D"))
        assert "UNBOUNDED" in joined.attrs["omnisea_aggregation"]["v@X"]

    def test_a_node_without_a_time_coordinate_warns_rather_than_vanishing(self, caplog):
        import logging

        odd = pd.DataFrame(
            {"v": [1.0, 2.0]},
            index=pd.DatetimeIndex(["2024-07-01", "2024-07-02"], name="obs_time"),
        )
        match = StationMatch(source="t", provider="t", station_id="O", name="O",
                             lat=48.8, lon=-125.1)
        tree = build_tree(
            Query.from_sites([BAMFIELD], WEEK),
            [StationSeries(match=match, frame=odd, node_path="in_situ/a/O",
                           attrs={}, var_attrs={})],
        )
        with caplog.at_level(logging.WARNING, logger="omnisea.align"):
            align(tree, freq="D")
        assert "no 'time' coordinate" in caplog.text

    def test_nodes_with_different_spans_share_one_index(self):
        """Resampling each series independently could hand back frames that do not line up."""
        q = Query.from_sites([BAMFIELD], WEEK)
        early = pd.date_range("2024-07-01", "2024-07-03", freq="h", tz="UTC", name="time")
        late = pd.date_range("2024-07-02", "2024-07-08", freq="h", tz="UTC", name="time")
        tree = build_tree(q, [
            series("A", "in_situ/a", pd.DataFrame({"va": range(len(early))}, index=early)),
            series("B", "in_situ/b", pd.DataFrame({"vb": range(len(late))}, index=late)),
        ])
        wide = align(tree, freq="h")
        assert wide.index.is_monotonic_increasing
        assert wide["va"].notna().any() and wide["vb"].notna().any()

    def test_calendar_frequencies_work(self):
        """'ME' has no fixed length, so a nanosecond-based grid could not describe it."""
        idx = pd.date_range("2024-07-01", "2024-09-30", freq="D", name="time")
        tree = self._tree(pd.DataFrame({"v": np.arange(float(len(idx)))}, index=idx))
        monthly = align(tree, freq="ME")
        assert len(monthly) == 3

    def test_an_all_empty_column_is_dropped_without_taking_others_with_it(self):
        idx = pd.date_range("2024-07-01", periods=5, freq="D", name="time")
        tree = self._tree(
            pd.DataFrame({"good": [1.0, 2, 3, 4, 5], "bad": [np.nan] * 5}, index=idx)
        )
        assert list(align(tree, freq="D").columns) == ["good"]


class TestCircularQuantities:
    """A compass bearing is not an ordinary number.

    350 deg and 10 deg both point nearly north; their arithmetic mean is 180 deg -- due south --
    and interpolating between them sweeps the long way round. Every source here publishes wind
    or wave directions, so getting this wrong is wrong everywhere at once.
    """

    def test_a_direction_is_recognised(self):
        from omnisea.align import is_circular

        assert is_circular({"units": "degree", "standard_name": "wind_from_direction"})
        assert is_circular({"units": "°", "standard_name": "sea_surface_wave_from_direction"})

    def test_a_directional_spread_is_not_circular(self):
        """A spread of 30 degrees is a width, and averages perfectly well."""
        from omnisea.align import is_circular

        assert not is_circular(
            {"units": "°", "standard_name": "sea_surface_wave_directional_spread"}
        )

    def test_a_plain_number_in_degrees_celsius_is_not_circular(self):
        from omnisea.align import is_circular

        assert not is_circular({"units": "degC", "standard_name": "air_temperature"})

    def test_directions_use_a_circular_mean_and_are_never_interpolated(self):
        assert aggregation_for(
            {"units": "degree", "standard_name": "wind_from_direction"},
            var="wind_from_direction",
        ) == ("circular_mean", "ffill")

    def test_the_circular_mean_of_north_readings_points_north(self):
        from omnisea.align import circular_mean

        def bearing_gap(a, b):
            return abs((a - b + 180.0) % 360.0 - 180.0)

        assert bearing_gap(circular_mean([350.0, 10.0]), 0.0) < 0.01
        assert bearing_gap(circular_mean([355.0, 5.0, 15.0]), 5.0) < 0.01

    def test_the_arithmetic_mean_would_have_pointed_south(self):
        """Pins the bug this rule exists for."""
        assert np.mean([350.0, 10.0]) == pytest.approx(180.0)

    def test_downsampling_a_direction_through_north_stays_north(self):
        q = Query.from_sites([BAMFIELD], WEEK)
        idx = pd.date_range("2024-07-01", periods=4, freq="15min", tz="UTC", name="time")
        frame = pd.DataFrame({"wind_from_direction": [350.0, 355.0, 5.0, 10.0]}, index=idx)
        tree = build_tree(q, [series("A", "in_situ/weather", frame, {
            "wind_from_direction": {"units": "degree", "standard_name": "wind_from_direction"}
        })])
        hourly = align(tree, freq="1h")
        got = hourly["wind_from_direction"].dropna().iloc[0]
        assert got > 350 or got < 10, f"expected a northerly bearing, got {got}"

    def test_the_rule_is_recorded_in_the_audit_trail(self):
        q = Query.from_sites([BAMFIELD], WEEK)
        idx = pd.date_range("2024-07-01", periods=8, freq="15min", tz="UTC", name="time")
        frame = pd.DataFrame({"wind_from_direction": [350.0] * 8}, index=idx)
        tree = build_tree(q, [series("A", "in_situ/weather", frame, {
            "wind_from_direction": {"units": "degree", "standard_name": "wind_from_direction"}
        })])
        assert "circular" in align(tree, freq="1h").attrs["omnisea_aggregation"][
            "wind_from_direction@A"
        ]


# --------------------------------------------------------------------------- redundancy


def wide_frame(n=40, seed=7):
    """A model matrix with a planted near-duplicate cluster and one independent column.

    temp_mean / temp_max / degree_days are the shape real daily climate takes: one signal
    published three ways. tide is genuinely independent of all of them.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-07-01", periods=n, freq="2h", name="time")
    base = np.sin(np.arange(n) / 5.0) * 4 + 15
    return pd.DataFrame({
        "temp_mean": base + rng.normal(0, 0.05, n),
        "temp_max": base + 3 + rng.normal(0, 0.05, n),
        "degree_days": (18 - base) + rng.normal(0, 0.05, n),  # strongly NEGATIVE correlate
        "tide": np.sin(np.arange(n) / 1.9) + 2 + rng.normal(0, 0.3, n),
    }, index=idx)


class TestCorrelations:
    def test_a_planted_pair_is_found_with_r_and_n(self):
        pairs = omnisea.correlations(wide_frame(), threshold=0.9)
        pair_sets = zip(pairs["feature_a"], pairs["feature_b"], strict=True)
        found = {frozenset((a, b)) for a, b in pair_sets}
        assert frozenset(("temp_mean", "temp_max")) in found
        row = pairs[(pairs["feature_a"] == "temp_mean") & (pairs["feature_b"] == "temp_max")]
        assert row["r"].iloc[0] > 0.99
        assert row["n"].iloc[0] == 40

    def test_strong_negative_correlation_ranks_by_magnitude(self):
        """degree_days runs opposite to temperature; |r| is what redundancy is about."""
        pairs = omnisea.correlations(wide_frame(), threshold=0.9)
        assert (pairs["r"] < 0).any(), "the negative correlate must be reported"
        magnitudes = pairs["r"].abs().to_list()
        assert magnitudes == sorted(magnitudes, reverse=True), "strongest pair first"

    def test_threshold_zero_shows_every_pair(self):
        pairs = omnisea.correlations(wide_frame(), threshold=0.0)
        assert len(pairs) == 6  # 4 columns -> 6 pairs

    def test_the_independent_column_is_not_flagged(self):
        pairs = omnisea.correlations(wide_frame(), threshold=0.9)
        flagged = set(pairs["feature_a"]) | set(pairs["feature_b"])
        assert "tide" not in flagged

    def test_a_perfect_r_over_three_points_is_not_reported(self):
        """Ragged overlap: r=1.0 on 3 samples is noise wearing a convincing costume."""
        frame = wide_frame()
        sparse = pd.Series(np.nan, index=frame.index)
        sparse.iloc[:3] = frame["temp_mean"].iloc[:3] * 2 + 1  # exactly collinear, n=3
        frame["sparse_sensor"] = sparse
        pairs = omnisea.correlations(frame, threshold=0.9)
        flagged = set(pairs["feature_a"]) | set(pairs["feature_b"])
        assert "sparse_sensor" not in flagged
        assert "sparse_sensor" in set(
            omnisea.correlations(frame, threshold=0.9, min_overlap=3)["feature_b"]
        ) | set(omnisea.correlations(frame, threshold=0.9, min_overlap=3)["feature_a"])

    def test_directions_and_qc_flags_are_excluded_but_spread_is_not(self):
        """Pearson r between bearings is meaningless; a directional SPREAD is a width."""
        frame = wide_frame()
        frame["wind_from_direction"] = frame["temp_mean"] * 20 % 360
        frame["wave_directional_spread"] = frame["temp_mean"] + 5
        frame["temp_mean_qc"] = 1.0
        pairs = omnisea.correlations(frame, threshold=0.9)
        flagged = set(pairs["feature_a"]) | set(pairs["feature_b"])
        assert "wind_from_direction" not in flagged
        assert "temp_mean_qc" not in flagged
        assert "wave_directional_spread" in flagged

    def test_fewer_than_two_usable_columns_is_an_empty_table(self):
        lone = pd.DataFrame({"only": [1.0, 2.0, 3.0]})
        pairs = omnisea.correlations(lone)
        assert list(pairs.columns) == ["feature_a", "feature_b", "r", "n"]
        assert pairs.empty

    def test_it_is_exported(self):
        from omnisea.align import correlations, drop_correlated

        assert omnisea.correlations is correlations
        assert omnisea.drop_correlated is drop_correlated


class TestDropCorrelated:
    def test_one_of_each_near_duplicate_pair_goes_and_the_reason_is_recorded(self):
        pruned = omnisea.drop_correlated(wide_frame(), threshold=0.95)
        cluster = {"temp_mean", "temp_max", "degree_days"}
        survivors = cluster & set(pruned.columns)
        assert len(survivors) == 1, f"the cluster must collapse to one column, kept {survivors}"
        assert "tide" in pruned.columns
        assert set(pruned.attrs["omnisea_dropped"]) == cluster - survivors
        for reason in pruned.attrs["omnisea_dropped"].values():
            assert "|r|=" in reason and "samples" in reason

    def test_the_better_covered_column_is_the_one_kept(self):
        frame = wide_frame()
        frame.loc[frame.index[:10], "temp_mean"] = np.nan  # temp_max now has more coverage
        pruned = omnisea.drop_correlated(frame[["temp_mean", "temp_max"]], threshold=0.95)
        assert list(pruned.columns) == ["temp_max"]

    def test_on_equal_coverage_the_first_column_wins(self):
        pruned = omnisea.drop_correlated(wide_frame()[["temp_mean", "temp_max"]], threshold=0.95)
        assert list(pruned.columns) == ["temp_mean"]

    def test_keep_pins_a_column_and_its_partner_goes_instead(self):
        frame = wide_frame()
        frame.loc[frame.index[:10], "temp_max"] = np.nan  # coverage says drop temp_max...
        pruned = omnisea.drop_correlated(
            frame[["temp_mean", "temp_max"]], threshold=0.95, keep="temp_max"
        )
        assert "temp_max" in pruned.columns  # ...but the pin overrides coverage
        assert "temp_mean" in pruned.attrs["omnisea_dropped"]

    def test_nothing_above_threshold_leaves_the_frame_alone(self):
        frame = wide_frame()[["temp_mean", "tide"]]
        pruned = omnisea.drop_correlated(frame, threshold=0.95)
        assert list(pruned.columns) == ["temp_mean", "tide"]
        assert pruned.attrs["omnisea_dropped"] == {}

    def test_existing_audit_attrs_survive_the_prune(self):
        frame = wide_frame()
        frame.attrs["omnisea_aggregation"] = {"temp_mean": "mean"}
        pruned = omnisea.drop_correlated(frame, threshold=0.95)
        assert pruned.attrs["omnisea_aggregation"] == {"temp_mean": "mean"}

    def test_your_own_columns_are_never_dropped_nor_cause_drops(self, ragged_tree):
        """End to end through align(on=...): y correlating with a feature is the POINT."""
        stamps = pd.date_range("2024-07-01", "2024-07-07", freq="6h")
        tree_frame = align(ragged_tree, on=pd.DataFrame({"time": stamps}), tolerance="4h")
        mine = pd.DataFrame({
            "time": stamps,
            # y is (nearly) the water level itself: r ~ 1 with a fetched feature.
            "my_response": tree_frame["water_level"].to_numpy() * 2 + 0.001,
        })
        data = align(ragged_tree, on=mine, tolerance="4h")
        assert "my_response" in data.attrs["omnisea_carried"]

        pruned = omnisea.drop_correlated(data, threshold=0.9)
        assert "my_response" in pruned.columns, "your response variable must survive"
        assert "water_level" in pruned.columns, (
            "correlation with YOUR column is signal, not redundancy — it must not "
            "knock the feature out"
        )


class TestReopenedTree:
    def test_align_on_your_timestamps_works_after_a_netcdf_round_trip(
        self, ragged_tree, tmp_path
    ):
        """Save today, model tomorrow: the reopened tree carries microsecond datetimes while
        a user's stamps are pandas-native nanoseconds, and merge_asof refuses to convert."""
        pytest.importorskip("netCDF4")
        path = tmp_path / "roundtrip.nc"
        ragged_tree.to_netcdf(path)
        reopened = xr.open_datatree(path)

        stamps = pd.date_range("2024-07-02", "2024-07-06", freq="6h")
        aligned = align(reopened, on=pd.DataFrame({"time": stamps}), tolerance="4h")
        assert aligned["water_level"].notna().all()
