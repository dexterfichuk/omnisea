"""Tree assembly, the netCDF-safety conversions, and the reader helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from omnisea.providers.base import StationMatch, StationSeries
from omnisea.query import Query, Site
from omnisea.tree import build_tree, coverage, series_to_dataset, stations, summary, to_dataframe

WEEK = ("2024-07-01", "2024-07-08")
BAMFIELD = Site(48.8353, -125.1358, "Bamfield")
VICTORIA = Site(48.42, -123.37, "Victoria")


def make_series(station_id="08545", name="Bamfield", node="in_situ/tides", n=6,
                site=None, lat=48.8353, lon=-125.1358, values=None):
    index = pd.date_range("2024-07-01", periods=n, freq="h", tz="UTC", name="time")
    frame = pd.DataFrame(
        {
            "water_surface_height_above_reference_datum": values
            if values is not None
            else np.linspace(1.0, 2.0, n),
            "water_surface_height_above_reference_datum_qc": ["1"] * n,
            "reviewed": [True] * n,
        },
        index=index,
    )
    match = StationMatch(
        source="dfo_tides", provider="dfo", station_id=station_id, name=name,
        lat=lat, lon=lon, site=site, distance_km=0.5,
    )
    return StationSeries(
        match=match,
        frame=frame,
        node_path=f"{node}/{station_id}",
        attrs={"Conventions": "CF-1.10", "provider": "dfo", "featureType": "timeSeries",
               "datum": "chart datum (CD)"},
        var_attrs={"water_surface_height_above_reference_datum": {
            "standard_name": "water_surface_height_above_reference_datum", "units": "m"}},
    )


class TestNodeStructure:
    def test_series_becomes_a_node_at_its_declared_path(self):
        q = Query.from_position(48.8353, -125.1358, WEEK)
        tree = build_tree(q, [make_series()])
        assert tree["/in_situ/tides/08545"].dataset["time"].size == 6

    def test_station_identity_is_stored_as_coordinates_not_attributes(self):
        """CF timeSeries wants lat/lon/station_id as coordinate variables."""
        ds = series_to_dataset(make_series())
        for coord in ("latitude", "longitude", "station_id", "station_name"):
            assert coord in ds.coords
        assert ds["station_id"].attrs["cf_role"] == "timeseries_id"

    def test_query_is_recorded_on_the_root(self):
        q = Query.from_position(48.8353, -125.1358, WEEK, radius_km=30)
        tree = build_tree(q, [make_series()])
        assert tree.attrs["query_start"].startswith("2024-07-01")
        assert tree.attrs["query_site_names"] == "Bamfield" or "48.8353" in str(
            tree.attrs["query_site_names"]
        )

    def test_predictions_and_observations_do_not_collide(self):
        q = Query.from_position(48.8353, -125.1358, WEEK)
        tree = build_tree(
            q,
            [make_series(node="in_situ/tides"), make_series(node="predictions/tides_hilo")],
        )
        assert "/in_situ/tides/08545" in tree.groups
        assert "/predictions/tides_hilo/08545" in tree.groups

    def test_duplicate_node_paths_are_suffixed_never_overwritten(self):
        q = Query.from_position(48.8353, -125.1358, WEEK)
        tree = build_tree(q, [make_series(), make_series()])
        data_nodes = [n.path for n in tree.subtree if n.dataset.data_vars]
        assert len(data_nodes) == 2


class TestEmptyResults:
    def test_no_results_gives_an_empty_tree_not_an_error(self):
        q = Query.from_position(48.8353, -125.1358, WEEK)
        tree = build_tree(q, [])
        assert isinstance(tree, xr.DataTree)
        assert not [n for n in tree.subtree if n.dataset.data_vars]
        assert tree.attrs["n_nodes"] == 0

    def test_station_with_no_rows_in_window_is_dropped(self):
        """A gauge whose record does not cover the dates should not appear as an empty group."""
        q = Query.from_position(48.8353, -125.1358, WEEK)
        empty = make_series()
        empty.frame = pd.DataFrame()
        tree = build_tree(q, [empty, make_series(station_id="08585")])
        assert tree.attrs["n_empty_series_dropped"] == 1
        assert "/in_situ/tides/08585" in tree.groups

    def test_summary_of_an_empty_tree_is_an_empty_frame(self):
        q = Query.from_position(48.8353, -125.1358, WEEK)
        assert summary(build_tree(q, [])).empty

    def test_to_dataframe_of_an_empty_tree_has_the_expected_columns(self):
        q = Query.from_position(48.8353, -125.1358, WEEK)
        frame = to_dataframe(build_tree(q, []))
        assert frame.empty
        assert "variable" in frame.columns


class TestNetcdfSafety:
    def test_time_is_stored_naive_because_cf_puts_the_zone_in_units(self):
        ds = series_to_dataset(make_series())
        assert ds["time"].dtype.kind == "M"
        assert not str(ds["time"].dtype).endswith("UTC]")
        assert ds["time"].attrs["time_zone"] == "UTC"

    def test_booleans_become_int8(self):
        ds = series_to_dataset(make_series())
        assert ds["reviewed"].dtype == np.int8

    def test_round_trip_through_netcdf_preserves_groups_and_values(self, tmp_path):
        pytest.importorskip("netCDF4")
        q = Query.from_position(48.8353, -125.1358, WEEK)
        tree = build_tree(
            q, [make_series(), make_series(station_id="08585", node="predictions/tides_hilo")]
        )
        path = tmp_path / "bamfield.nc"
        tree.to_netcdf(path)
        back = xr.open_datatree(path)
        assert {n.path for n in tree.subtree if n.dataset.data_vars} == {
            n.path for n in back.subtree if n.dataset.data_vars
        }
        np.testing.assert_allclose(
            tree["/in_situ/tides/08545"].dataset[
                "water_surface_height_above_reference_datum"].values,
            back["/in_situ/tides/08545"].dataset[
                "water_surface_height_above_reference_datum"].values,
        )

    def test_attributes_survive_the_round_trip(self, tmp_path):
        pytest.importorskip("netCDF4")
        q = Query.from_position(48.8353, -125.1358, WEEK)
        tree = build_tree(q, [make_series()])
        path = tmp_path / "attrs.nc"
        tree.to_netcdf(path)
        back = xr.open_datatree(path)
        assert back["/in_situ/tides/08545"].attrs["datum"] == "chart datum (CD)"


class TestMultiSite:
    def test_site_label_is_carried_as_a_coordinate(self):
        q = Query.from_sites([BAMFIELD, VICTORIA], WEEK)
        tree = build_tree(q, [make_series(site="Bamfield")])
        assert tree["/in_situ/tides/08545"].dataset["site"].item() == "Bamfield"

    def test_group_by_site_nests_nodes_under_their_location(self):
        q = Query.from_sites([BAMFIELD, VICTORIA], WEEK)
        tree = build_tree(
            q,
            [
                make_series(site="Bamfield"),
                make_series(station_id="07120", site="Victoria", lat=48.42, lon=-123.37),
            ],
            group_by_site=True,
        )
        assert "/Bamfield/in_situ/tides/08545" in tree.groups
        assert "/Victoria/in_situ/tides/07120" in tree.groups

    def test_coverage_lists_sites_that_found_nothing(self):
        """With a long list of locations, the empty ones are the result you most need."""
        q = Query.from_sites([BAMFIELD, VICTORIA], WEEK)
        tree = build_tree(q, [make_series(site="Bamfield")])
        cov = coverage(tree, q).set_index("site")
        assert bool(cov.loc["Bamfield", "has_data"])
        assert not bool(cov.loc["Victoria", "has_data"])
        assert int(cov.loc["Victoria", "n_time"]) == 0

    def test_site_labels_with_awkward_characters_are_made_node_safe(self):
        q = Query.from_sites([Site(48.8353, -125.1358, "Bamfield / Inlet #2")], WEEK)
        tree = build_tree(q, [make_series(site="Bamfield / Inlet #2")], group_by_site=True)
        assert any("Bamfield" in g for g in tree.groups)


class TestReaders:
    def test_summary_reports_one_row_per_node(self):
        q = Query.from_position(48.8353, -125.1358, WEEK)
        tree = build_tree(q, [make_series(), make_series(station_id="08585")])
        frame = summary(tree)
        assert len(frame) == 2
        assert set(frame.columns) >= {"node", "provider", "station_id", "n_time", "start", "end"}

    def test_summary_excludes_qc_columns_from_the_variable_list(self):
        q = Query.from_position(48.8353, -125.1358, WEEK)
        listed = summary(build_tree(q, [make_series()])).iloc[0]["variables"]
        assert "_qc" not in listed

    def test_to_dataframe_is_long_and_carries_the_join_keys(self):
        q = Query.from_sites([BAMFIELD], WEEK)
        frame = to_dataframe(build_tree(q, [make_series(site="Bamfield")]))
        assert {"time", "variable", "value", "site", "station_id"} <= set(frame.columns)
        assert (frame["site"] == "Bamfield").all()

    def test_to_dataframe_wide_keeps_one_column_per_variable(self):
        q = Query.from_position(48.8353, -125.1358, WEEK)
        frame = to_dataframe(build_tree(q, [make_series()]), wide=True)
        assert "water_surface_height_above_reference_datum" in frame.columns

    def test_stations_collapses_nodes_to_one_row_per_station(self):
        q = Query.from_position(48.8353, -125.1358, WEEK)
        tree = build_tree(
            q, [make_series(node="in_situ/tides"), make_series(node="predictions/tides_hilo")]
        )
        frame = stations(tree)
        assert len(frame) == 1
        assert frame.iloc[0]["n_nodes"] == 2
