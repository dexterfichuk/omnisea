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
