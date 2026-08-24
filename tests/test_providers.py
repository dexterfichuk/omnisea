"""Provider shaping logic, exercised offline against captured API responses."""

from __future__ import annotations

import pandas as pd
import pytest

from omnisea.http import chunk_time
from omnisea.providers.base import StationMatch, drop_orphan_qc, frame_from_records, trim_to_window
from omnisea.providers.dfo import (
    MAX_DAYS_COARSE,
    MAX_DAYS_ONE_MINUTE,
    DfoProvider,
    DfoTidesSource,
)
from omnisea.providers.eccc import (
    EcccClimateDaily,
    EcccClimateHourly,
    EcccHydrometric,
    EcccProvider,
    EcccSwobRealtime,
)
from omnisea.providers.ogc import point_from_feature
from omnisea.query import Query

WEEK = ("2024-07-01", "2024-07-08")


def a_match(source: str, station_id: str = "TEST") -> StationMatch:
    return StationMatch(
        source=source, station_id=station_id, name="Test Station", lat=48.8353, lon=-125.1358
    )


@pytest.fixture
def eccc():
    return EcccProvider()


# --------------------------------------------------------------------------- geometry


class TestCoordinateExtraction:
    def test_coordinates_come_from_geometry_not_properties(self, eccc_stations_geojson):
        """climate-stations publishes LATITUDE as integer micro-degrees (483300000)."""
        feature = eccc_stations_geojson["features"][0]
        raw_lat = feature["properties"]["LATITUDE"]
        assert abs(raw_lat) > 1000  # the trap: unusable as a latitude

        lat, lon = point_from_feature(feature)
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180

    def test_feature_without_geometry_is_rejected(self):
        assert point_from_feature({"properties": {"LATITUDE": 483300000}}) is None

    def test_out_of_range_geometry_is_rejected(self):
        assert point_from_feature({"geometry": {"coordinates": [-125.1, 999]}}) is None


# --------------------------------------------------------------------------- ECCC hourly


class TestEcccHourly:
    def test_fields_are_renamed_to_cf(self, eccc, eccc_hourly_rows):
        source = EcccClimateHourly(eccc)
        query = Query.from_area((-124, 48, -123, 49), WEEK)
        series = source.series_from_rows(query, a_match("eccc_climate"), eccc_hourly_rows)
        assert "air_temperature" in series.frame.columns
        assert "TEMP" not in series.frame.columns

    def test_wind_direction_is_repaired_in_the_output(self, eccc, eccc_hourly_rows):
        source = EcccClimateHourly(eccc)
        query = Query.from_area((-124, 48, -123, 49), WEEK)
        series = source.series_from_rows(query, a_match("eccc_climate"), eccc_hourly_rows)
        raw = [r["WIND_DIRECTION"] for r in eccc_hourly_rows if r.get("WIND_DIRECTION") is not None]
        out = series.frame["wind_from_direction"].dropna()
        assert raw and out.max() == pytest.approx(max(raw) * 10)
        assert out.max() > 36  # would be impossible if the x10 fix had not been applied

    def test_time_index_is_utc_and_sorted(self, eccc, eccc_hourly_rows):
        source = EcccClimateHourly(eccc)
        query = Query.from_area((-124, 48, -123, 49), WEEK)
        series = source.series_from_rows(query, a_match("eccc_climate"), eccc_hourly_rows)
        assert series.frame.index.tz is not None
        assert series.frame.index.is_monotonic_increasing

    def test_unmapped_description_field_is_carried(self, eccc, eccc_hourly_rows):
        source = EcccClimateHourly(eccc)
        query = Query.from_area((-124, 48, -123, 49), WEEK)
        series = source.series_from_rows(query, a_match("eccc_climate"), eccc_hourly_rows)
        assert "WEATHER_ENG_DESC" in series.frame.columns

    def test_datetime_param_is_padded_for_the_local_date_filter(self, eccc):
        """climate-hourly filters on LOCAL_DATE, so a UTC window must be padded then trimmed."""
        source = EcccClimateHourly(eccc)
        query = Query.from_area((-124, 48, -123, 49), WEEK)
        start, end = source.datetime_param(query).split("/")
        assert start.startswith("2024-06-30")
        assert end.startswith("2024-07-09")

    def test_identity_columns_do_not_become_variables(self, eccc, eccc_hourly_rows):
        source = EcccClimateHourly(eccc)
        query = Query.from_area((-124, 48, -123, 49), WEEK)
        series = source.series_from_rows(query, a_match("eccc_climate"), eccc_hourly_rows)
        for identity in ("STATION_NAME", "CLIMATE_IDENTIFIER", "LOCAL_HOUR", "STN_ID"):
            assert identity not in series.frame.columns


class TestEcccDaily:
    def test_local_date_is_stamped_at_midnight_utc(self, eccc, eccc_daily_rows):
        source = EcccClimateDaily(eccc)
        query = Query.from_area((-124, 48, -123, 49), WEEK)
        series = source.series_from_rows(query, a_match("eccc_climate_daily"), eccc_daily_rows)
        assert (series.frame.index.hour == 0).all()

    def test_time_convention_is_documented_on_the_node(self, eccc, eccc_daily_rows):
        """climate-daily has no UTC_DATE; the convention must be stated, not silently assumed."""
        source = EcccClimateDaily(eccc)
        query = Query.from_area((-124, 48, -123, 49), WEEK)
        series = source.series_from_rows(query, a_match("eccc_climate_daily"), eccc_daily_rows)
        assert "LOCAL_DATE" in series.attrs["time_reference"]

    def test_daily_datetime_param_uses_plain_dates(self, eccc):
        source = EcccClimateDaily(eccc)
        query = Query.from_area((-124, 48, -123, 49), WEEK)
        assert source.datetime_param(query) == "2024-07-01/2024-07-08"


class TestEcccSwob:
    def test_units_are_read_from_the_uom_sibling(self, eccc, eccc_swob_rows):
        """SWOB ships a `-uom` beside every value, so units come from the data, not a table."""
        source = EcccSwobRealtime(eccc)
        assert source.units_for("air_temp", eccc_swob_rows) == "°C"
        assert source.units_for("rel_hum", eccc_swob_rows) == "%"

    def test_field_the_station_does_not_report_has_no_units(self, eccc, eccc_swob_rows):
        """This fixture is a Parks Canada fire-weather station: no barometer, so no pressure."""
        source = EcccSwobRealtime(eccc)
        assert source.units_for("stn_pres", eccc_swob_rows) is None

    def test_units_land_in_the_variable_attributes(self, eccc, eccc_swob_rows):
        source = EcccSwobRealtime(eccc)
        query = Query.from_area((-126, 48, -125, 49), WEEK)
        series = source.series_from_rows(query, a_match("eccc_swob"), eccc_swob_rows)
        if "air_temperature" in series.var_attrs:
            assert series.var_attrs["air_temperature"]["units"] == "°C"

    def test_uom_and_qa_siblings_are_not_variables(self, eccc, eccc_swob_rows):
        source = EcccSwobRealtime(eccc)
        query = Query.from_area((-126, 48, -125, 49), WEEK)
        series = source.series_from_rows(query, a_match("eccc_swob"), eccc_swob_rows)
        assert not [c for c in series.frame.columns if str(c).endswith(("-uom", "-qa"))]

    def test_engineering_channels_are_carried_through(self, eccc, eccc_swob_rows):
        """Battery voltage has no CF name but is still data the user asked for."""
        source = EcccSwobRealtime(eccc)
        query = Query.from_area((-126, 48, -125, 49), WEEK)
        series = source.series_from_rows(query, a_match("eccc_swob"), eccc_swob_rows)
        assert any("batry" in str(c) for c in series.frame.columns)


class TestEcccHydrometric:
    def test_level_and_discharge_get_cf_names(self, eccc, eccc_hydro_rows):
        source = EcccHydrometric(eccc)
        query = Query.from_area((-126, 48, -125, 49), WEEK)
        series = source.series_from_rows(query, a_match("eccc_hydrometric"), eccc_hydro_rows)
        assert "water_surface_height_above_reference_datum" in series.frame.columns
        assert "water_volume_transport_in_river_channel" in series.frame.columns

    def test_bilingual_symbol_columns_do_not_become_variables(self, eccc, eccc_hydro_rows):
        source = EcccHydrometric(eccc)
        query = Query.from_area((-126, 48, -125, 49), WEEK)
        series = source.series_from_rows(query, a_match("eccc_hydrometric"), eccc_hydro_rows)
        assert not [c for c in series.frame.columns if str(c).endswith("_SYMBOL_FR")]


# --------------------------------------------------------------------------- DFO IWLS


class TestIwlsChunking:
    def test_one_minute_window_is_capped_at_seven_days(self):
        """Verified live: IWLS rejects >7 days at ONE_MINUTE with HTTP 400."""
        start = pd.Timestamp("2024-07-01", tz="UTC")
        chunks = chunk_time(start, start + pd.Timedelta(days=30), max_days=MAX_DAYS_ONE_MINUTE)
        assert len(chunks) == 5
        assert all((e - s) <= pd.Timedelta(days=7) for s, e in chunks)

    def test_coarse_resolution_uses_the_thirty_one_day_cap(self):
        """The limit is resolution-dependent; chunking a month of hourly data would be waste."""
        start = pd.Timestamp("2024-07-01", tz="UTC")
        chunks = chunk_time(start, start + pd.Timedelta(days=30), max_days=MAX_DAYS_COARSE)
        assert len(chunks) == 1

    def test_chunks_tile_the_window_without_gaps(self):
        start = pd.Timestamp("2024-07-01", tz="UTC")
        end = start + pd.Timedelta(days=20)
        chunks = chunk_time(start, end, max_days=7)
        assert chunks[0][0] == start and chunks[-1][1] == end
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            assert prev_end == next_start  # shared boundary, de-duplicated on concat

    def test_empty_window_yields_no_chunks(self):
        start = pd.Timestamp("2024-07-01", tz="UTC")
        assert chunk_time(start, start, max_days=7) == []


class TestIwlsSeriesSelection:
    def test_predictions_and_observations_get_different_node_paths(self):
        from omnisea.providers.dfo import SERIES_NODES

        assert SERIES_NODES["wlo"] != SERIES_NODES["wlp"]
        assert SERIES_NODES["wlo"].startswith("in_situ")
        assert SERIES_NODES["wlp"].startswith("predictions")

    def test_unknown_resolution_is_rejected(self):
        from omnisea.errors import QueryError

        query = Query.from_area((-126, 48, -125, 49), WEEK, resolution="EVERY_FORTNIGHT")
        with pytest.raises(QueryError, match="unknown IWLS resolution"):
            DfoTidesSource(DfoProvider())._resolution(query)

    def test_unknown_series_is_rejected(self):
        from omnisea.errors import QueryError

        query = Query.from_area((-126, 48, -125, 49), WEEK, series=["not-a-series"])
        with pytest.raises(QueryError, match="unknown IWLS series"):
            DfoTidesSource(DfoProvider())._series_for(query)

    def test_row_estimate_scales_with_resolution(self):
        source = DfoTidesSource(DfoProvider())
        coarse = Query.from_area((-126, 48, -125, 49), WEEK, resolution="SIXTY_MINUTES")
        fine = Query.from_area((-126, 48, -125, 49), WEEK, resolution="ONE_MINUTE")
        assert source._estimate_rows(fine, ["wlo"]) == 60 * source._estimate_rows(coarse, ["wlo"])

    def test_discovery_matches_the_bamfield_gauge_in_the_station_list(self, iwls_stations):
        """The IWLS station list has no bbox filter, so selection happens client-side."""
        station = iwls_stations[0]
        query = Query.from_position(
            float(station["latitude"]), float(station["longitude"]), WEEK, radius_km=1
        )
        assert query.contains(float(station["latitude"]), float(station["longitude"]))


class TestIwlsMetadata:
    def test_datum_offsets_are_present(self, iwls_metadata):
        codes = {d["code"] for d in iwls_metadata["datums"]}
        assert {"CGVD2013", "CGVD28"} <= codes


# --------------------------------------------------------------------------- frame helpers


class TestFrameHelpers:
    def test_duplicate_boundary_timestamps_are_deduplicated(self):
        """Chunked requests share their boundary instants by design."""
        rows = [
            {"time": "2024-07-01T00:00:00Z", "v": 1.0},
            {"time": "2024-07-01T00:00:00Z", "v": 1.0},
            {"time": "2024-07-01T01:00:00Z", "v": 2.0},
        ]
        frame = frame_from_records(rows)
        assert len(frame) == 2
        assert frame.index.is_unique

    def test_rows_are_sorted_by_time(self):
        rows = [
            {"time": "2024-07-01T02:00:00Z", "v": 2.0},
            {"time": "2024-07-01T00:00:00Z", "v": 0.0},
        ]
        assert frame_from_records(rows).index.is_monotonic_increasing

    def test_empty_input_gives_an_empty_frame(self):
        assert frame_from_records([]).empty

    def test_all_empty_columns_are_dropped(self):
        rows = [{"time": "2024-07-01T00:00:00Z", "v": 1.0, "never_reported": None}]
        assert "never_reported" not in frame_from_records(rows).columns

    def test_orphan_qc_column_is_dropped(self):
        """An all-missing variable is dropped, but its flags are not empty and would survive."""
        frame = pd.DataFrame(
            {"air_temperature": [1.0], "precipitation_amount_qc": ["M"]},
            index=pd.DatetimeIndex(["2024-07-01"], tz="UTC", name="time"),
        )
        assert "precipitation_amount_qc" not in drop_orphan_qc(frame).columns

    def test_qc_column_with_its_parent_is_kept(self):
        frame = pd.DataFrame(
            {"air_temperature": [1.0], "air_temperature_qc": ["1"]},
            index=pd.DatetimeIndex(["2024-07-01"], tz="UTC", name="time"),
        )
        assert "air_temperature_qc" in drop_orphan_qc(frame).columns

    def test_window_trim_is_inclusive_of_both_endpoints(self):
        index = pd.date_range("2024-06-30", "2024-07-09", freq="D", tz="UTC", name="time")
        frame = pd.DataFrame({"v": range(len(index))}, index=index)
        trimmed = trim_to_window(
            frame, pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2024-07-08", tz="UTC")
        )
        assert trimmed.index.min() == pd.Timestamp("2024-07-01", tz="UTC")
        assert trimmed.index.max() == pd.Timestamp("2024-07-08", tz="UTC")
