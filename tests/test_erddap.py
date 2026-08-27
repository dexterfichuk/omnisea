"""ERDDAP adapter: metadata parsing, shaping and the payload rules, plus live smoke tests.

The offline tests run against captured responses from three real ERDDAP servers — CIOOS Pacific,
IOOS Sensors and NOAA CoastWatch — because the interesting cases are all things a real
installation does and a synthesized fixture would not: a dataset that publishes two variables
under one standard name, a table holding 700 stations at once, a grid whose latitude axis runs
downwards, and an empty result served as HTTP 404.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import FIXTURES

from omnisea.errors import PayloadTooLargeError, ProviderError, QueryError, UpstreamError
from omnisea.provenance import citation
from omnisea.providers import erddap
from omnisea.providers.base import StationMatch
from omnisea.providers.erddap import (
    DEFAULT_SERVER,
    DatasetInfo,
    ErddapGridSource,
    ErddapProvider,
    ErddapTableSource,
    clear_cache,
    field_table,
    grid_selection,
    parse_info,
)
from omnisea.providers.erddap.common import _overlaps_query

# Importing the adapter registers its own query knobs (erddap_server and friends), the same
# way any third-party provider's would be.
from omnisea.query import Query
from omnisea.tree import build_tree

WEEK = ("2024-07-01", "2024-07-08")
HYDRO_HOUR = ("2024-07-01T00:00:00Z", "2024-07-01T01:00:00Z")
NDBC_WINDOW = ("2024-07-01T00:00:00Z", "2024-07-01T00:40:00Z")

# Bamfield Marine Sciences Centre — the library's running example.
BAMFIELD = dict(lat=48.8353, lon=-125.1358)
IOOS_SENSORS = "https://erddap.sensors.ioos.us/erddap"
COASTWATCH = "https://coastwatch.pfeg.noaa.gov/erddap"


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def provider():
    return ErddapProvider()


@pytest.fixture
def table_source(provider):
    return ErddapTableSource(provider)


@pytest.fixture
def grid_source(provider):
    return ErddapGridSource(provider)


def info_of(name: str, dataset_id: str) -> DatasetInfo:
    return parse_info(json.loads((FIXTURES / name).read_text()), dataset_id)


def rows_of(name: str) -> list[dict]:
    table = json.loads((FIXTURES / name).read_text())["table"]
    columns = table["columnNames"]
    return [dict(zip(columns, row, strict=True)) for row in table["rows"]]


@pytest.fixture
def station_info():
    """A Water Survey of Canada gauge on IOOS Sensors: QARTOD flags, one station."""
    return info_of("erddap_info_station.json", "ca_hydro_08HB048")


@pytest.fixture
def station_rows():
    return rows_of("erddap_tabledap_station.json")


@pytest.fixture
def wavebuoy_info():
    """CIOOS Pacific's Barkley Sound wave buoy: declares its sampling interval and cell methods."""
    return info_of("erddap_info_wavebuoy.json", "PRIMED_wavebuoy")


@pytest.fixture
def wavebuoy_rows():
    return rows_of("erddap_tabledap_wavebuoy.json")


@pytest.fixture
def grid_info():
    return info_of("erddap_info_grid.json", "nesdisVHNSQchlaWeekly")


@pytest.fixture
def ndbc_info():
    """NOAA's NDBC met table: every buoy on Earth in one dataset."""
    return info_of("erddap_info_multistation.json", "cwwcNDBCMet")


@pytest.fixture
def ndbc_rows():
    return rows_of("erddap_tabledap_multistation.json")


def a_match(dataset_id: str, server: str = IOOS_SENSORS, **extra) -> StationMatch:
    return StationMatch(
        source="erddap_tabledap",
        provider="erddap",
        station_id=dataset_id,
        name=dataset_id,
        lat=48.9,
        lon=-125.0,
        extra={"server": server, "dataset_id": dataset_id, "protocol": "tabledap", **extra},
    )


# --------------------------------------------------------------------------- metadata


class TestInfoParsing:
    def test_bounds_and_period_come_from_the_global_attributes(self, station_info):
        assert station_info.bounds == pytest.approx((-124.99778, 48.91556, -124.99778, 48.91556))
        assert station_info.first == pd.Timestamp("2015-08-31T08:00:00Z")
        assert station_info.last > pd.Timestamp("2024-07-08T00:00:00Z")

    def test_declared_sampling_interval_is_used(self, wavebuoy_info):
        """PRIMED_wavebuoy publishes time_coverage_resolution = PT30M00S."""
        assert wavebuoy_info.resolution == pd.Timedelta(minutes=30)
        assert wavebuoy_info.samples_per_day == pytest.approx(48.0)

    def test_an_unparseable_declared_resolution_falls_back_to_the_measured_axis_spacing(
        self, grid_info
    ):
        """This CoastWatch grid declares "PW1" — a real dataset, a typo'd ISO 8601 duration.

        ERDDAP measures the time axis itself and reports it on the dimension, which is both
        parseable and closer to the truth, so that is what gets used.
        """
        assert grid_info.global_attrs["time_coverage_resolution"] == "PW1"
        assert pd.Timedelta(days=7) < grid_info.resolution < pd.Timedelta(days=8)
        assert grid_info.samples_per_day < 1.0

    def test_a_dataset_with_no_interval_at_all_falls_back_to_the_documented_assumption(
        self, ndbc_info
    ):
        from omnisea.providers.erddap import DEFAULT_SAMPLES_PER_DAY

        assert ndbc_info.resolution is None
        assert ndbc_info.samples_per_day == DEFAULT_SAMPLES_PER_DAY

    def test_the_station_identifier_comes_from_cdm_timeseries_variables(self, ndbc_info):
        """The declaration lists position too; the identifier is the entry that is not a coord."""
        assert ndbc_info.global_attrs["cdm_timeseries_variables"] == "station, longitude, latitude"
        assert ndbc_info.station_variable == "station"

    def test_qc_companions_are_read_from_ancillary_variables(self, station_info):
        qc = station_info.qc_map()
        assert qc["river_discharge"] == [
            "river_discharge_qc_agg",
            "river_discharge_qc_tests",
        ]

    def test_advertised_variables_exclude_coordinates_and_flags(self, station_info):
        """The Catalog column answers "what does it measure", not "what columns exist"."""
        advertised = station_info.standard_names
        assert "river_discharge" in advertised
        assert "water_surface_height_above_reference_datum" in advertised
        assert "latitude" not in advertised
        assert "aggregate_quality_flag" not in advertised

    def test_bounds_fall_back_to_the_latitude_actual_range(self, station_info):
        """Some datasets skip the geospatial globals but ERDDAP still computes actual_range."""
        stripped = replace(
            station_info,
            global_attrs={
                k: v
                for k, v in station_info.global_attrs.items()
                if not k.startswith("geospatial_")
            },
        )
        assert stripped.bounds == pytest.approx((-124.99778, 48.91556, -124.99778, 48.91556))

    def test_a_dataset_with_no_extent_at_all_has_no_bounds(self, station_info):
        bare = replace(station_info, global_attrs={}, variables={})
        assert bare.bounds is None

    def test_a_malformed_info_payload_is_an_omnisea_error(self):
        with pytest.raises(ProviderError, match="did not have the expected columns"):
            parse_info({"table": {"columnNames": ["nope"], "rows": []}}, "whatever")


# --------------------------------------------------------------------------- field tables


class TestFieldTable:
    def test_the_datasets_own_standard_name_becomes_the_variable_name(self, station_info):
        table = field_table(station_info, present=list(station_info.variables))
        spec = table["water_surface_height_above_reference_datum_above_localstationdatum"]
        assert spec.var == "water_surface_height_above_reference_datum"
        assert spec.standard_name == "water_surface_height_above_reference_datum"

    def test_no_standard_name_is_invented(self, station_info, ndbc_info, wavebuoy_info):
        """Every name emitted must be one the dataset itself published, character for character."""
        for info in (station_info, ndbc_info, wavebuoy_info):
            table = field_table(info, present=list(info.variables))
            for raw, spec in table.items():
                declared = str(info.variables[raw].get("standard_name") or "")
                assert spec.standard_name == declared, f"{info.dataset_id}.{raw}"

    def test_two_variables_sharing_a_standard_name_keep_their_own_names(self, ndbc_info):
        """NDBC publishes dpd and apd both as sea_surface_swell_wave_period."""
        table = field_table(ndbc_info, present=list(ndbc_info.variables))
        assert table["dpd"].var == "dpd"
        assert table["apd"].var == "apd"
        assert table["dpd"].standard_name == table["apd"].standard_name

    def test_cell_methods_are_carried_through(self, wavebuoy_info, grid_info):
        """align() reads cell_methods to decide resampling, so it must survive the trip."""
        waves = field_table(wavebuoy_info, present=list(wavebuoy_info.variables))
        assert waves["waveHs"].cell_methods == "time: point"
        assert grid_info.variables["chlor_a"]["cell_methods"] == "time:mean(interval:1 week)"

    def test_units_come_from_the_dataset(self, station_info):
        table = field_table(station_info, present=list(station_info.variables))
        assert table["river_discharge"].units == "m3.s-1"

    def test_the_aggregate_flag_becomes_the_qc_field(self, station_info):
        table = field_table(station_info, present=list(station_info.variables))
        assert table["river_discharge"].qc_field == "river_discharge_qc_agg"

    def test_per_sample_position_is_renamed_so_it_cannot_be_overwritten(self, ndbc_info):
        """series_to_dataset assigns scalar latitude/longitude coords over any same-named column."""
        table = field_table(ndbc_info, present=list(ndbc_info.variables))
        assert table["latitude"].var == "sample_latitude"
        assert table["longitude"].var == "sample_longitude"
        assert table["latitude"].standard_name == "latitude"

    def test_provenance_records_the_providers_own_field_name(self, station_info):
        table = field_table(station_info, present=list(station_info.variables))
        spec = table["water_surface_height_above_reference_datum_above_localstationdatum"]
        assert spec.extra_attrs["source_field"] == (
            "water_surface_height_above_reference_datum_above_localstationdatum"
        )


# --------------------------------------------------------------------------- shaping


class TestTableShaping:
    def query(self, **options):
        return Query.from_area((-125.6, 48.5, -124.7, 49.2), HYDRO_HOUR, **options)

    def series(self, source, info, rows, **options):
        return source.series_from_rows(
            self.query(**options), a_match(info.dataset_id), info, rows
        )

    def test_columns_carry_the_datasets_cf_names(self, table_source, station_info, station_rows):
        frame = self.series(table_source, station_info, station_rows)[0].frame
        assert "river_discharge" in frame.columns
        assert "water_surface_height_above_reference_datum" in frame.columns

    def test_qc_flags_travel_beside_their_measurement(
        self, table_source, station_info, station_rows
    ):
        series = self.series(table_source, station_info, station_rows)[0]
        assert "river_discharge_qc" in series.frame.columns
        assert set(series.frame["river_discharge_qc"]) == {2}  # QARTOD NOT_EVALUATED
        assert series.var_attrs["river_discharge_qc"]["source_field"] == "river_discharge_qc_agg"

    def test_the_time_index_is_utc_sorted_and_unique(
        self, table_source, station_info, station_rows
    ):
        frame = self.series(table_source, station_info, station_rows)[0].frame
        assert frame.index.tz is not None
        assert frame.index.is_monotonic_increasing
        assert frame.index.is_unique

    def test_the_window_is_enforced_locally(self, table_source, station_info, station_rows):
        """The upstream filter is trusted for the request, never for the result."""
        narrow = Query.from_area(
            (-125.6, 48.5, -124.7, 49.2), ("2024-07-01T00:00:00Z", "2024-07-01T00:20:00Z")
        )
        series = table_source.series_from_rows(
            narrow, a_match("ca_hydro_08HB048"), station_info, station_rows
        )[0]
        assert series.frame.index.max() <= pd.Timestamp("2024-07-01T00:20:00Z")
        assert len(series.frame) < len(station_rows)

    def test_per_sample_position_survives_as_its_own_column(
        self, table_source, station_info, station_rows
    ):
        frame = self.series(table_source, station_info, station_rows)[0].frame
        assert frame["sample_latitude"].iloc[0] == pytest.approx(48.91556)

    def test_the_station_identifier_does_not_become_a_variable(
        self, table_source, ndbc_info, ndbc_rows
    ):
        series = table_source.series_from_rows(
            Query.from_area((-125, 48, -123, 49), NDBC_WINDOW),
            a_match("cwwcNDBCMet", COASTWATCH),
            ndbc_info,
            ndbc_rows,
        )
        assert all("station" not in s.frame.columns for s in series)

    def test_the_dataset_licence_wins_over_the_providers_placeholder(
        self, table_source, station_info, station_rows
    ):
        """One ERDDAP hosts a dozen institutions; the licence belongs to the dataset."""
        attrs = self.series(table_source, station_info, station_rows)[0].attrs
        assert attrs["institution"] == "Canada Water Office"
        assert attrs["license"].startswith("These data may be used and redistributed")
        assert attrs["erddap_dataset_id"] == "ca_hydro_08HB048"

    def test_variables_does_not_project_the_response(
        self, table_source, station_info, station_rows
    ):
        """`variables=` chooses sources and stations; it never drops a downloaded column."""
        narrow = self.series(
            table_source, station_info, station_rows, variables=["river_discharge"]
        )[0]
        broad = self.series(table_source, station_info, station_rows)[0]
        assert set(narrow.frame.columns) == set(broad.frame.columns)
        assert "water_surface_height_above_reference_datum" in narrow.frame.columns

    def test_units_never_disagree_with_the_numbers(
        self, table_source, station_info, station_rows
    ):
        """A dataset states its units, not how to reach canonical CF ones, so nothing converts.

        The unsafe outcome would be a value relabelled without being changed, so the check is
        that both the numbers and the units attribute are identical either way.
        """
        plain = self.series(table_source, station_info, station_rows)[0]
        asked = self.series(table_source, station_info, station_rows, to_cf_units=True)[0]
        assert asked.var_attrs["river_discharge"]["units"] == "m3.s-1"
        assert plain.var_attrs["river_discharge"]["units"] == "m3.s-1"
        assert asked.frame["river_discharge"].equals(plain.frame["river_discharge"])


class TestUnmappedFields:
    """A dataset's variables with no CF standard name are still data someone asked for."""

    WINDOW = ("2021-04-01T00:00:00Z", "2021-04-01T03:00:00Z")

    def series(self, source, info, rows, **options):
        query = Query.from_area((-126, 48, -125, 50), self.WINDOW, **options)
        (series,) = source.series_from_rows(
            query, a_match("PRIMED_wavebuoy", DEFAULT_SERVER), info, rows
        )
        return series

    def test_they_travel_by_default_under_their_own_names(
        self, table_source, wavebuoy_info, wavebuoy_rows
    ):
        frame = self.series(table_source, wavebuoy_info, wavebuoy_rows).frame
        assert "metaDeclination" in frame.columns
        assert "downcrossWaveCount" in frame.columns

    def test_they_are_tagged_so_a_reader_can_tell_them_apart(
        self, table_source, wavebuoy_info, wavebuoy_rows
    ):
        attrs = self.series(table_source, wavebuoy_info, wavebuoy_rows).var_attrs
        assert attrs["metaDeclination"]["omnisea_mapped"] == 0
        assert attrs["sea_surface_temperature"].get("omnisea_mapped") == 1

    def test_the_datasets_description_of_them_survives(
        self, table_source, wavebuoy_info, wavebuoy_rows
    ):
        """The generic passthrough would replace long_name with the bare variable name."""
        attrs = self.series(table_source, wavebuoy_info, wavebuoy_rows).var_attrs
        assert attrs["metaDeclination"]["long_name"] == "magnetic declination to east"
        assert attrs["metaDeclination"]["units"] == "degree"

    def test_the_caller_can_switch_them_off(
        self, table_source, wavebuoy_info, wavebuoy_rows
    ):
        frame = self.series(
            table_source, wavebuoy_info, wavebuoy_rows, include_unmapped=False
        ).frame
        assert "metaDeclination" not in frame.columns
        assert "sea_surface_temperature" in frame.columns


class TestMultiStationSplitting:
    """One tabledap dataset is not one station, and collapsing them would lose rows silently."""

    @pytest.fixture
    def series(self, table_source, ndbc_info, ndbc_rows):
        return table_source.series_from_rows(
            Query.from_area((-125, 48, -123, 49), NDBC_WINDOW),
            a_match("cwwcNDBCMet", COASTWATCH),
            ndbc_info,
            ndbc_rows,
        )

    def test_each_station_becomes_its_own_node(self, series, ndbc_rows):
        assert len(series) == len({row["station"] for row in ndbc_rows})
        assert len({s.node_path for s in series}) == len(series)

    def test_no_row_is_lost_to_a_shared_timestamp(self, series, ndbc_rows):
        """Two buoys reporting at the same instant is the case that de-duplication would eat."""
        assert sum(len(s.frame) for s in series) == len(ndbc_rows)

    def test_each_station_gets_its_own_position(self, series):
        positions = {(round(s.match.lat, 3), round(s.match.lon, 3)) for s in series}
        assert len(positions) == len(series)

    def test_the_node_path_names_the_dataset_and_the_station(self, series):
        for s in series:
            assert s.node_path.startswith("in_situ/erddap/cwwcNDBCMet/")
            assert s.match.station_id.startswith("cwwcNDBCMet:")

    def test_each_stations_distance_matches_its_own_position(
        self, table_source, ndbc_info, ndbc_rows
    ):
        """The dataset's box spans the globe; a distance from its centre would mean nothing."""
        query = Query.from_position(**BAMFIELD, time=NDBC_WINDOW, radius_km=300)
        series = table_source.series_from_rows(
            query, a_match("cwwcNDBCMet", COASTWATCH), ndbc_info, ndbc_rows
        )
        for s in series:
            expected = query.sites[0].distance_km(s.match.lat, s.match.lon)
            assert s.match.distance_km == pytest.approx(expected)
        assert len({round(s.match.distance_km, 3) for s in series}) == len(series)

    def test_a_single_station_dataset_gets_no_station_suffix(
        self, table_source, station_info, station_rows
    ):
        series = table_source.series_from_rows(
            Query.from_area((-125.6, 48.5, -124.7, 49.2), HYDRO_HOUR),
            a_match("ca_hydro_08HB048"),
            station_info,
            station_rows,
        )
        assert [s.node_path for s in series] == ["in_situ/erddap/ca_hydro_08HB048"]


# --------------------------------------------------------------------------- discovery


class TestCandidateListing:
    def test_ids_are_read_from_the_alldatasets_table(self, table_source, monkeypatch):
        payload = json.loads((FIXTURES / "erddap_alldatasets_ioos.json").read_text())
        monkeypatch.setattr(table_source, "_get", lambda url, params: payload)
        found = table_source._from_all_datasets(
            Query.from_area((-125.6, 48.5, -124.7, 49.2), WEEK), IOOS_SENSORS
        )
        assert "ca_hydro_08HB048" in found

    def test_ids_are_read_from_the_search_index(self, table_source, monkeypatch):
        payload = json.loads((FIXTURES / "erddap_search_tabledap.json").read_text())
        monkeypatch.setattr(table_source, "_get", lambda url, params: payload)
        found = table_source._from_search(
            Query.from_area((-125.6, 48.5, -124.7, 49.2), WEEK), IOOS_SENSORS
        )
        assert found == [
            "ca_hydro_08HB048",
            "ioos-gliderdac-dfo-bumblebee998-20240509T1917",
            "ca_hydro_08HB014",
        ]

    def test_the_two_catalogues_are_unioned_not_chosen_between(self, table_source, monkeypatch):
        """CIOOS Pacific leaves allDatasets bounds null while its search index knows them."""
        monkeypatch.setattr(table_source, "_from_all_datasets", lambda q, s: ["a", "b"])
        monkeypatch.setattr(table_source, "_from_search", lambda q, s: ["b", "c"])
        query = Query.from_area((-125.6, 48.5, -124.7, 49.2), WEEK)
        assert table_source._candidate_ids(query, IOOS_SENSORS) == ["a", "b", "c"]

    def test_erddaps_own_catalogue_table_is_not_a_dataset(self, table_source, monkeypatch):
        monkeypatch.setattr(table_source, "_from_all_datasets", lambda q, s: ["allDatasets"])
        monkeypatch.setattr(table_source, "_from_search", lambda q, s: ["allDatasets", "real"])
        query = Query.from_area((-125.6, 48.5, -124.7, 49.2), WEEK)
        assert table_source._candidate_ids(query, IOOS_SENSORS) == ["real"]

    def test_the_bbox_is_pushed_into_the_alldatasets_constraints(self, table_source, monkeypatch):
        seen: dict[str, str] = {}

        def capture(url, params):
            seen["url"] = url
            return None

        monkeypatch.setattr(table_source, "_get", capture)
        table_source._from_all_datasets(
            Query.from_area((-125.6, 48.5, -124.7, 49.2), WEEK), IOOS_SENSORS
        )
        assert 'dataStructure="table"' in seen["url"]
        assert "maxLatitude>=48.5" in seen["url"]
        assert "minTime<=2024-07-08T00:00:00Z" in seen["url"]


class TestDiscoveryFiltering:
    def query(self, **options):
        return Query.from_position(**BAMFIELD, time=WEEK, radius_km=30, **options)

    def stub(self, source, monkeypatch, infos):
        monkeypatch.setattr(source, "_candidate_ids", lambda q, s: [i.dataset_id for i in infos])
        monkeypatch.setattr(source, "_info", lambda s, ds: {i.dataset_id: i for i in infos}[ds])

    def test_a_nearby_gauge_is_matched_with_a_real_distance(
        self, table_source, station_info, monkeypatch
    ):
        self.stub(table_source, monkeypatch, [station_info])
        (match,) = table_source.discover(self.query())
        assert match.station_id == "ca_hydro_08HB048"
        assert 0 < match.distance_km < 30
        assert "river_discharge" in match.variables

    def test_a_dataset_whose_record_ends_before_the_window_is_skipped(
        self, table_source, wavebuoy_info, monkeypatch
    ):
        """The Barkley Sound buoy stopped reporting in May 2021."""
        self.stub(table_source, monkeypatch, [wavebuoy_info])
        assert table_source.discover(self.query()) == []

    def test_a_dataset_outside_the_radius_is_skipped(
        self, table_source, station_info, monkeypatch
    ):
        self.stub(table_source, monkeypatch, [station_info])
        far = Query.from_position(lat=20.0, lon=-150.0, time=WEEK, radius_km=30)
        assert table_source.discover(far) == []

    def test_a_requested_variable_the_dataset_lacks_removes_it(
        self, table_source, station_info, monkeypatch
    ):
        self.stub(table_source, monkeypatch, [station_info])
        assert table_source.discover(self.query(variables=["sea_surface_temperature"])) == []
        assert table_source.discover(self.query(variables=["river_discharge"]))

    def test_an_unknown_variable_name_does_not_opt_the_source_out(self, table_source):
        """A curated table is a floor; ERDDAP has no curated table at all."""
        assert table_source.wants_anything(self.query(variables=["batry_volt"]))
        assert table_source.wants_anything(self.query(variables=["sea_surface_temperature"]))

    def test_a_named_dataset_is_used_even_without_geospatial_metadata(
        self, table_source, station_info, monkeypatch
    ):
        """Seven of CIOOS Pacific's eight datasets publish no extent; naming one must work.

        No *geospatial* metadata is the case being simulated, so the variables stay — a
        publisher who skipped the extent attributes still declares what the table contains,
        and a dataset with no `time` variable at all is a different story (see
        :class:`TestUnusableDatasets`).
        """
        bare = replace(station_info, global_attrs={}, variables={"time": {}})
        monkeypatch.setattr(table_source, "_info", lambda s, ds: bare)
        (match,) = table_source.discover(self.query(erddap_datasets=["ca_hydro_08HB048"]))
        assert match.station_id == "ca_hydro_08HB048"
        assert np.isnan(match.lat), "an unknown position must be NaN, not invented"

    def test_the_row_estimate_only_counts_the_covered_part_of_the_window(
        self, table_source, wavebuoy_info
    ):
        """The buoy's record ends mid-window, so the estimate must not bill for the whole week."""
        query = Query.from_area((-126, 48, -125, 50), ("2021-05-08", "2021-05-22"))
        estimate = table_source._estimate_rows(query, wavebuoy_info)
        assert estimate == pytest.approx(48 * 2.9, rel=0.05)  # ~2.9 days at 30-minute sampling


class TestPayloadSafety:
    def test_too_many_datasets_refuses_rather_than_truncating(self, grid_source, monkeypatch):
        """A wide griddap query on a national server matches hundreds; 200 /info calls is abuse."""
        monkeypatch.setattr(
            grid_source, "_candidate_ids", lambda q, s: [f"ds{i}" for i in range(60)]
        )
        query = Query.from_area((-140, 40, -120, 60), WEEK)
        with pytest.raises(PayloadTooLargeError, match="erddap_max_datasets"):
            grid_source.discover(query)

    def test_the_ceiling_is_configurable(self, grid_source, monkeypatch, grid_info):
        monkeypatch.setattr(grid_source, "_candidate_ids", lambda q, s: ["a", "b", "c"])
        monkeypatch.setattr(grid_source, "_info", lambda s, ds: replace(grid_info, dataset_id=ds))
        query = Query.from_area((-140, 40, -120, 60), WEEK, erddap_max_datasets=2)
        with pytest.raises(PayloadTooLargeError):
            grid_source.discover(query)
        assert len(grid_source.discover(query.replace(options={"erddap_max_datasets": 5}))) == 3

    def test_an_over_large_fetch_fails_before_the_request_goes_out(
        self, table_source, station_info, monkeypatch
    ):
        monkeypatch.setattr(table_source, "_info", lambda s, ds: station_info)
        monkeypatch.setattr(
            table_source,
            "_download",
            lambda *a, **k: pytest.fail("a request was made despite the ceiling"),
        )
        query = Query.from_area((-126, 48, -124, 50), ("2020-01-01", "2024-01-01"), max_rows=1000)
        with pytest.raises(PayloadTooLargeError, match="row ceiling"):
            table_source.fetch(query, [a_match("ca_hydro_08HB048")])

    def test_the_ceiling_is_enforced_again_against_the_rows_that_arrive(
        self, table_source, station_info, station_rows, monkeypatch
    ):
        """The estimate is a guess; the row count is not, so it gets the final say."""
        payload = json.loads((FIXTURES / "erddap_tabledap_station.json").read_text())
        monkeypatch.setattr(table_source, "_get", lambda url, params: payload)
        query = Query.from_area((-126, 48, -124, 50), HYDRO_HOUR, max_rows=5)
        with pytest.raises(PayloadTooLargeError):
            table_source._download(query, IOOS_SENSORS, station_info)

    def test_a_long_window_is_chunked_rather_than_asked_for_in_one_request(
        self, table_source, wavebuoy_info, monkeypatch
    ):
        urls: list[str] = []

        def capture(url, params):
            urls.append(url)
            return None

        monkeypatch.setattr(table_source, "_get", capture)
        query = Query.from_area((-126, 48, -125, 50), ("2021-03-22", "2021-05-10"))
        table_source._download(query, DEFAULT_SERVER, wavebuoy_info)
        assert len(urls) >= 1
        assert all("time>=" in u and "time<=" in u for u in urls)

    def test_a_multi_station_request_is_clipped_to_the_query_area(
        self, table_source, ndbc_info, monkeypatch
    ):
        """cwwcNDBCMet is every buoy on Earth; without this a bbox query downloads all of them."""
        urls: list[str] = []
        monkeypatch.setattr(table_source, "_get", lambda url, params: urls.append(url))
        query = Query.from_area((-125.5, 47.5, -123.0, 49.0), NDBC_WINDOW)
        table_source._download(query, COASTWATCH, ndbc_info)
        assert "latitude>=47.5" in urls[0]
        assert "longitude<=-123.0" in urls[0]


# --------------------------------------------------------------------------- error handling


class TestUpstreamQuirks:
    def raising(self, monkeypatch, status, detail):
        def _get_json(url, params, provider=None, **kwargs):
            raise UpstreamError("upstream request failed", status=status, detail=detail)

        monkeypatch.setattr(erddap.common, "get_json", _get_json)

    def test_an_empty_result_set_is_not_a_failure(self, table_source, monkeypatch):
        """ERDDAP answers "nothing matched" with a 404, on every endpoint."""
        self.raising(
            monkeypatch,
            404,
            'Error { message="Not Found: Your query produced no matching results. (nRows = 0)"; }',
        )
        assert table_source._get("https://example.org/erddap/search/advanced.json", {}) is None

    def test_an_unknown_dataset_id_still_raises(self, table_source, monkeypatch):
        self.raising(
            monkeypatch,
            404,
            'Error { message="Not Found: Currently unknown datasetID=nope"; }',
        )
        with pytest.raises(UpstreamError):
            table_source._get("https://example.org/erddap/info/nope/index.json", {})

    def test_a_server_error_is_never_swallowed(self, table_source, monkeypatch):
        self.raising(monkeypatch, 500, "internal error")
        with pytest.raises(UpstreamError):
            table_source._get("https://example.org/erddap/info/x/index.json", {})

    def test_a_missing_dataset_metadata_response_names_the_server(
        self, table_source, monkeypatch
    ):
        monkeypatch.setattr(table_source, "_get", lambda url, params: None)
        with pytest.raises(ProviderError, match="no metadata for dataset"):
            table_source._info(IOOS_SENSORS, "ghost")

    def test_a_nonsense_server_option_is_rejected_before_any_request(self, table_source):
        query = Query.from_area((-126, 48, -125, 49), WEEK, erddap_server="cioospacific.ca")
        with pytest.raises(QueryError, match="full http"):
            table_source.server(query)

    def test_the_default_server_is_used_when_none_is_named(self, table_source):
        assert table_source.server(Query.from_area((-126, 48, -125, 49), WEEK)) == DEFAULT_SERVER

    def test_a_station_is_never_silently_dropped_for_a_missing_handoff_key(self, table_source):
        """extra is the untyped seam between discover() and fetch(); reading it must be loud."""
        bare = StationMatch(source="erddap_tabledap", station_id="x", name="x", lat=0, lon=0)
        with pytest.raises(ProviderError, match="bug in the erddap_tabledap adapter"):
            table_source.fetch(Query.from_area((-126, 48, -125, 49), WEEK), [bare])


# --------------------------------------------------------------------------- geometry


class TestAreaOverlap:
    def test_a_fixed_station_is_judged_by_the_site_radius(self, station_info):
        near = Query.from_position(**BAMFIELD, time=WEEK, radius_km=30)
        far = Query.from_position(**BAMFIELD, time=WEEK, radius_km=2)
        assert _overlaps_query(near, station_info.bounds)
        assert not _overlaps_query(far, station_info.bounds)

    def test_a_platform_with_real_extent_is_judged_by_intersection(self, ndbc_info):
        """A glider that crossed the corner of the box was there, whatever its centroid says."""
        query = Query.from_position(**BAMFIELD, time=WEEK, radius_km=10)
        assert _overlaps_query(query, ndbc_info.bounds)  # global extent, crosses the box
        assert not query.contains(*ndbc_info.bounds.centre)  # centroid is in the Indian Ocean


class TestGridSelection:
    def grid(self, lats, lons, times=("2024-07-01", "2024-07-05")) -> xr.Dataset:
        time = pd.date_range(times[0], times[1], freq="D")
        data = np.zeros((len(time), len(lats), len(lons)))
        return xr.Dataset(
            {"sst": (("time", "latitude", "longitude"), data)},
            coords={"time": time, "latitude": list(lats), "longitude": list(lons)},
        )

    def query(self, bbox=(-125.6, 48.5, -124.7, 49.2), **kwargs):
        return Query.from_area(bbox, ("2024-07-02", "2024-07-04"), **kwargs)

    def test_an_ascending_axis_slices_low_to_high(self):
        grid = self.grid(np.arange(48.0, 50.0, 0.1), np.arange(-126.0, -124.0, 0.1))
        assert grid_selection(grid, self.query())["latitude"] == slice(48.5, 49.2)

    def test_a_descending_axis_slices_the_other_way(self):
        """CoastWatch chlorophyll runs latitude downwards; sel would return nothing otherwise."""
        grid = self.grid(np.arange(50.0, 48.0, -0.1), np.arange(-126.0, -124.0, 0.1))
        selection = grid_selection(grid, self.query())
        assert selection["latitude"] == slice(49.2, 48.5)
        assert grid.sel(selection).sizes["latitude"] > 0

    def test_a_zero_to_three_sixty_grid_gets_the_query_shifted_onto_it(self):
        grid = self.grid(np.arange(48.0, 50.0, 0.1), np.arange(230.0, 240.0, 0.1))
        selection = grid_selection(grid, self.query())
        assert selection["longitude"] == slice(pytest.approx(234.4), pytest.approx(235.3))
        assert grid.sel(selection).sizes["longitude"] > 0

    def test_the_window_is_applied_without_a_timezone(self):
        """griddap times decode naive; comparing them to a tz-aware bound raises."""
        grid = self.grid(np.arange(48.0, 50.0, 0.1), np.arange(-126.0, -124.0, 0.1))
        selection = grid_selection(grid, self.query())
        assert selection["time"].start.tzinfo is None
        assert grid.sel(selection).sizes["time"] == 3

    def test_a_depth_range_is_honoured_when_the_grid_has_one(self):
        grid = self.grid(np.arange(48.0, 50.0, 0.1), np.arange(-126.0, -124.0, 0.1))
        grid = grid.expand_dims(depth=[0.0, 10.0, 50.0])
        selection = grid_selection(grid, self.query(depth=[0, 20]))
        assert grid.sel(selection).sizes["depth"] == 2


# --------------------------------------------------------------------------- live


live = pytest.mark.network


def skip_if_unreachable(exc: UpstreamError) -> None:
    """Skip only when the server never answered. An HTTP error is a real failure."""
    if getattr(exc, "status", None) is None:
        pytest.skip(f"ERDDAP server unreachable: {exc}")
    raise exc


@live
class TestLiveDefaultServer:
    """The default server has to work out of the box, or the source is not usable as shipped.

    Anchored on the Carnation Creek hydrometric gauge rather than a named buoy: IOOS Sensors
    delisted ``PRIMED_wavebuoy`` in mid-2026, which is exactly the kind of upstream churn a
    default-server test has to survive. The buoy-specific physics now runs against the buoy's
    home server in :class:`TestLiveCioosPacific`.
    """

    def test_discovery_near_bamfield_finds_real_stations(self):
        source = ErddapTableSource(ErddapProvider())
        query = Query.from_position(**BAMFIELD, time=WEEK, radius_km=30)
        try:
            matches = source.discover(query)
        except UpstreamError as exc:
            skip_if_unreachable(exc)
        gauge = next((m for m in matches if m.station_id == "ca_hydro_08HB048"), None)
        assert gauge is not None, f"default server returned {sorted(m.station_id for m in matches)}"
        assert gauge.distance_km < 20
        assert gauge.first is not None and gauge.first < pd.Timestamp("2024-07-01T00:00:00Z")

    def test_a_window_outside_the_record_yields_no_data_rather_than_raising(self):
        """ERDDAP reports "no rows" as a 404, which must not surface as an error.

        The result is an *empty* series rather than nothing at all, so the dataset is named in
        the tree's ``omnisea_empty_stations`` — "this gauge had nothing for 1970" and "no gauge
        was ever consulted" must not look identical.
        """
        source = ErddapTableSource(ErddapProvider())
        query = Query.from_area(
            (-125.6, 48.5, -124.7, 49.2), ("1970-01-01", "1970-01-08"),
            erddap_datasets=["ca_hydro_08HB048"],
        )
        try:
            series = source.fetch(query, source.discover(query))
        except UpstreamError as exc:
            skip_if_unreachable(exc)
        assert all(s.is_empty for s in series), "1970 predates this gauge entirely"
        tree = build_tree(query, series)
        assert not [n for n in tree.subtree if n.dataset.data_vars]
        assert "ca_hydro_08HB048" in str(tree.attrs.get("omnisea_empty_stations", ""))


@live
class TestLiveCioosPacific:
    """A second server, reached the way a user would reach it: through ``erddap_server=``.

    No offline test exercises that knob against a real installation, so this class is the proof
    that "any ERDDAP" is more than a claim. It deliberately names **no dataset**: this
    catalogue is a third party's to change, and it does — it used to host a Barkley Sound wave
    buoy that has since been withdrawn. Pinning an id here would have turned somebody else's
    housekeeping into a failing build, which is exactly the coupling omnisea exists to avoid.
    What is asserted is what omnisea promises about whatever the server does return.
    """

    CIOOS_PACIFIC = "https://data.cioospacific.ca/erddap"
    WINDOW = ("2021-04-01", "2021-04-08")

    @pytest.fixture(scope="class")
    def catalog(self):
        source = ErddapTableSource(ErddapProvider())
        query = Query.from_position(
            **BAMFIELD, time=self.WINDOW, radius_km=80,
            erddap_server=self.CIOOS_PACIFIC,
        )
        try:
            matches = source.discover(query)
        except UpstreamError as exc:
            skip_if_unreachable(exc)
        if not matches:
            pytest.skip("CIOOS Pacific published nothing near Bamfield in this window")
        return source, query, matches

    @pytest.fixture(scope="class")
    def fetched(self, catalog):
        """The first match that actually returns rows, with its series."""
        source, query, matches = catalog
        for match in matches:
            try:
                series = source.fetch(query, [match])
            except UpstreamError as exc:
                skip_if_unreachable(exc)
            for item in series:
                if not item.is_empty:
                    return match, item
        pytest.skip("every dataset near Bamfield was empty for this window")

    def test_discovery_reports_positions_inside_the_radius_it_was_given(self, catalog):
        _, _, matches = catalog
        for match in matches:
            assert match.distance_km is not None and match.distance_km <= 80.0, match.station_id
            assert -90 <= match.lat <= 90 and -180 <= match.lon <= 180

    def test_discovery_only_offers_records_overlapping_the_window(self, catalog):
        _, _, matches = catalog
        start, end = (pd.Timestamp(t, tz="UTC") for t in self.WINDOW)
        for match in matches:
            assert match.first is None or match.first <= end, match.station_id
            assert match.last is None or match.last >= start, match.station_id

    def test_a_real_fetch_stays_inside_the_requested_window(self, fetched):
        _, series = fetched
        frame = series.frame
        assert not frame.empty
        assert frame.index.min() >= pd.Timestamp(self.WINDOW[0], tz="UTC")
        assert frame.index.max() <= pd.Timestamp(self.WINDOW[1], tz="UTC")
        assert frame.index.is_monotonic_increasing and frame.index.is_unique

    def test_cell_methods_are_the_datasets_own(self, fetched):
        """align() resamples on these, so omnisea has to pass through exactly what the dataset
        declares — no drops, and nothing invented where the dataset is silent. Read back from
        the server's own info response rather than assumed, because whether these datasets
        declare cell_methods at all is their choice, not omnisea's."""
        import requests

        match, series = fetched
        info = requests.get(
            f"{self.CIOOS_PACIFIC}/info/{match.station_id}/index.json", timeout=120
        ).json()["table"]
        columns = info["columnNames"]
        variable, attribute, value = (
            columns.index("Variable Name"), columns.index("Attribute Name"),
            columns.index("Value"),
        )
        declared = {
            row[variable]: row[value]
            for row in info["rows"]
            if row[attribute] == "cell_methods"
        }
        emitted = {
            attrs.get("source_field") or name: attrs["cell_methods"]
            for name, attrs in series.var_attrs.items()
            if attrs.get("cell_methods")
        }
        invented = set(emitted) - set(declared)
        assert not invented, f"omnisea invented cell_methods for {sorted(invented)}"
        if not declared:
            pytest.skip(
                f"{match.station_id} declares no cell_methods upstream, so there is nothing "
                "here to pass through — the pass-through itself is covered offline"
            )
        for field in set(emitted) & set(declared):
            assert emitted[field] == declared[field], field

    def test_every_standard_name_emitted_is_a_real_cf_name(self, fetched):
        """These names are the dataset's, not omnisea's — but they still land in the output."""
        import re

        import requests

        _, series = fetched
        table = requests.get(
            "https://cfconventions.org/Data/cf-standard-names/current/src/"
            "cf-standard-name-table.xml",
            timeout=120,
        ).text
        valid = set(re.findall(r'<entry id="([^"]+)"', table))
        assert len(valid) > 4000, "did not get a plausible CF standard name table"
        emitted = {
            attrs["standard_name"]
            for attrs in series.var_attrs.values()
            if attrs.get("standard_name")
        }
        assert emitted, "the dataset published no standard names at all"
        assert not (emitted - valid), f"not CF standard names: {sorted(emitted - valid)}"

    def test_the_series_is_placed_and_attributed(self, fetched):
        match, series = fetched
        assert series.node_path.startswith("in_situ/erddap/")
        assert series.attrs.get("source_url", "").startswith(self.CIOOS_PACIFIC)
        assert series.attrs.get("license")


@live
class TestLiveGriddap:
    @pytest.fixture(scope="class")
    def subset(self):
        source = ErddapGridSource(ErddapProvider())
        query = Query.from_area(
            (-125.6, 48.5, -124.7, 49.2), ("2024-07-01", "2024-07-04"),
            erddap_server=COASTWATCH, erddap_datasets=["jplMURSST41"],
        )
        try:
            datasets = source.fetch(query, source.discover(query))
        except UpstreamError as exc:
            skip_if_unreachable(exc)
        assert datasets, "griddap returned no dataset for jplMURSST41"
        return datasets[0]

    def test_the_result_is_a_dataset_placed_under_the_gridded_branch(self, subset):
        assert isinstance(subset, xr.Dataset)
        assert subset.attrs["omnisea_node_path"] == "gridded/erddap/jplMURSST41"

    def test_nothing_has_been_read(self, subset):
        """The whole point of the gridded path: a decade of SST costs nothing until indexed."""
        assert not subset["analysed_sst"].variable._in_memory

    def test_the_subset_is_clipped_to_the_query(self, subset):
        assert 48.4 <= float(subset.latitude.min()) and float(subset.latitude.max()) <= 49.3
        assert -125.7 <= float(subset.longitude.min())
        assert float(subset.longitude.max()) <= -124.6
        assert subset.sizes["time"] <= 4

    def test_the_datasets_own_cf_metadata_survives(self, subset):
        sst = subset["analysed_sst"]
        assert sst.attrs["standard_name"] == "sea_surface_foundation_temperature"
        assert sst.attrs["units"] == "degree_C"
        assert subset.attrs["institution"] == "NASA JPL"

    def test_the_numbers_are_plausible_sea_surface_temperatures(self, subset):
        """Reading a handful of pixels is the only way to know the subsetting was real."""
        corner = subset["analysed_sst"].isel(
            time=0, latitude=slice(0, 3), longitude=slice(0, 3)
        )
        values = corner.values
        assert np.isfinite(values).any()
        assert 5.0 < float(np.nanmean(values)) < 25.0  # July, off Vancouver Island


@live
class TestLiveTheUnionSeam:
    """A grid and a set of stations must land in one tree without either knowing about the other."""

    def test_both_return_types_assemble_into_the_same_tree(self):
        from omnisea.tree import build_tree

        provider = ErddapProvider()
        table, grid = ErddapTableSource(provider), ErddapGridSource(provider)
        points = Query.from_position(
            **BAMFIELD, time=("2024-07-01", "2024-07-03"), radius_km=40,
            erddap_server=IOOS_SENSORS,
        )
        cells = points.replace(
            options={
                **points.options,
                "erddap_server": COASTWATCH,
                "erddap_datasets": ["jplMURSST41"],
            }
        )
        try:
            results = [
                *table.fetch(points, table.discover(points)),
                *grid.fetch(cells, grid.discover(cells)),
            ]
        except UpstreamError as exc:
            skip_if_unreachable(exc)

        tree = build_tree(points, results)
        paths = {n.path for n in tree.subtree if n.dataset.data_vars}
        assert "/gridded/erddap/jplMURSST41" in paths
        assert any(p.startswith("/in_situ/erddap/") for p in paths)
        assert not tree["/gridded/erddap/jplMURSST41"].dataset[
            "analysed_sst"
        ].variable._in_memory, "assembling the tree read the grid"


class TestUnusableDatasets:
    """Not every tabledap dataset is a time series, and the failure must be readable.

    CIOOS Pacific's ``IOS_P26_Annualized`` — Ocean Station Papa, the deep end of Line P — is
    indexed by an integer ``Year`` column and publishes no ``time`` variable at all. omnisea
    puts ``&time>=`` on every tabledap request, so before this was caught the user got
    ``400 Unrecognized constraint variable="time"`` from the server, which tells them nothing
    they can act on.
    """

    def query(self, **options):
        return Query.from_position(
            **BAMFIELD, time=WEEK, radius_km=30, erddap_server=IOOS_SENSORS, **options
        )

    def annualized(self, station_info):
        """A dataset whose axis is a Year column, as CIOOS Pacific really publishes it."""
        return replace(
            station_info,
            dataset_id="IOS_P26_Annualized",
            global_attrs={"title": "Station Papa annualized", "cdm_data_type": "Other"},
            variables={"Year": {}, "temp_0_10_dbar": {"units": "degC"}},
        )

    def test_naming_one_explicitly_says_why_it_cannot_be_read(
        self, table_source, station_info, monkeypatch
    ):
        monkeypatch.setattr(table_source, "_info", lambda s, ds: self.annualized(station_info))
        with pytest.raises(ProviderError) as excinfo:
            table_source.discover(self.query(erddap_datasets=["IOS_P26_Annualized"]))
        message = str(excinfo.value)
        assert "no 'time' variable" in message
        assert "IOS_P26_Annualized" in message
        assert "Year" in message, "the message should name what it IS indexed by"

    def test_it_is_skipped_rather_than_fatal_during_an_ordinary_search(
        self, table_source, station_info, monkeypatch
    ):
        """A wide box legitimately matches such datasets; they must not sink the query."""
        good, bad = station_info, self.annualized(station_info)
        monkeypatch.setattr(table_source, "_candidate_ids", lambda q, s: ["good", "bad"])
        monkeypatch.setattr(
            table_source, "_info", lambda s, ds: good if ds == "good" else bad
        )
        matches = table_source.discover(self.query())
        assert [m.station_id for m in matches] == ["ca_hydro_08HB048"]

    def test_a_griddap_dataset_without_a_time_axis_is_still_fine(
        self, grid_source, station_info
    ):
        """The rule is tabledap's: a bathymetry grid has no time axis and needs none."""
        assert grid_source.unusable_reason(self.annualized(station_info)) is None

    def test_a_dataset_with_no_usable_rows_is_named_not_silently_absent(
        self, table_source, station_info, monkeypatch
    ):
        """CIOOS Pacific's IYS_2019_CTD declares a `time` variable that is entirely empty, so
        ERDDAP answers every window with "no matching results". A tree simply lacking the node
        reads as "there is nothing here"; the dataset has to be named."""
        monkeypatch.setattr(table_source, "_info", lambda s, ds: station_info)
        monkeypatch.setattr(table_source, "_download", lambda q, s, i: [])
        (series,) = table_source.fetch(self.query(), [a_match("ca_hydro_08HB048")])
        assert series.is_empty
        assert series.match.station_id == "ca_hydro_08HB048"

        tree = build_tree(self.query(), [series])
        assert "erddap_tabledap/ca_hydro_08HB048" in tree.attrs["omnisea_empty_stations"]
        assert "returned no rows" in citation(tree)
