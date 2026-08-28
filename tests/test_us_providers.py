"""NOAA CO-OPS and USGS NWIS — the US partners to dfo_tides and eccc_hydrometric.

Fixtures are captured live responses, trimmed. What these guard: the branch symmetry with the
Canadian sources (observations never share a node with predictions), the datum being stated
where the ERDDAP mirror lost it, and the NWIS shapes — RDB catalogues, no-data sentinels,
qualifier codes — surviving translation.
"""

from __future__ import annotations

import pandas as pd
import pytest

import omnisea
from omnisea.providers import noaa as noaa_mod
from omnisea.providers import usgs as usgs_mod
from omnisea.providers.noaa import CoopsProvider, CoopsWaterSource
from omnisea.providers.usgs import UsgsProvider, UsgsWaterSource, _parse_rdb
from omnisea.query import Query, Site
from omnisea.tree import build_tree

WINDOW = ("2024-07-01", "2024-07-03")

# Captured from the live APIs on 2026-08-27, trimmed.
COOPS_STATIONS = {
    "stations": [
        {"id": "9444090", "name": "Port Angeles", "lat": 48.125, "lng": -123.44,
         "state": "WA", "greatlakes": False},
        {"id": "9087031", "name": "Holland", "lat": 42.7733, "lng": -86.2128,
         "state": "MI", "greatlakes": True},
    ]
}
COOPS_WL = {
    "metadata": {"id": "9444090", "name": "Port Angeles", "lat": "48.125", "lon": "-123.44"},
    "data": [
        {"t": "2024-07-01 00:00", "v": "1.241", "s": "0.009", "f": "0,0,0,0", "q": "v"},
        {"t": "2024-07-01 00:06", "v": "1.249", "s": "0.007", "f": "0,0,0,0", "q": "v"},
        {"t": "2024-07-01 00:12", "v": "", "s": "", "f": "0,0,0,0", "q": "p"},
    ],
}
COOPS_HILO = {
    "predictions": [
        {"t": "2024-07-01 05:36", "v": "2.311", "type": "H"},
        {"t": "2024-07-01 13:22", "v": "-0.126", "type": "L"},
    ]
}
NWIS_RDB = (
    "# comment\n"
    "agency_cd\tsite_no\tstation_nm\tsite_tp_cd\tdec_lat_va\tdec_long_va\tparm_cd\t"
    "data_type_cd\tbegin_date\tend_date\n"
    "5s\t15s\t50s\t7s\t16s\t16s\t5s\t2s\t10d\t10d\n"
    "USGS\t12045500\tELWHA RIVER AT MCDONALD BR\tST\t48.0547802\t-123.5832136\t00060\t"
    "uv\t1897-10-01\t2026-08-27\n"
    "USGS\t12045500\tELWHA RIVER AT MCDONALD BR\tST\t48.0547802\t-123.5832136\t00065\t"
    "uv\t2007-10-01\t2026-08-27\n"
    "USGS\t12045500\tELWHA RIVER AT MCDONALD BR\tST\t48.0547802\t-123.5832136\t00060\t"
    "dv\t1897-10-01\t2026-08-27\n"
    # A site holding ONLY water-quality grab samples: the IV service answers nothing for it.
    "USGS\t88888888\tQW ONLY CREEK\tST\t48.06\t-123.55\t00060\tqw\t2000-01-01\t2026-08-27\n"
    "USGS\t99999999\tLONG GONE CREEK\tST\t48.1\t-123.4\t00060\tuv\t1910-01-01\t1972-10-31\n"
)
NWIS_IV = {
    "value": {
        "timeSeries": [
            {
                "sourceInfo": {"siteName": "ELWHA RIVER AT MCDONALD BR"},
                "variable": {
                    "variableCode": [{"value": "00060"}],
                    "unit": {"unitCode": "ft3/s"},
                    "noDataValue": -999999.0,
                },
                "values": [{"value": [
                    {"value": "1000", "qualifiers": ["A"],
                     "dateTime": "2024-06-30T17:00:00.000-07:00"},
                    {"value": "-999999", "qualifiers": ["P"],
                     "dateTime": "2024-06-30T17:15:00.000-07:00"},
                ]}],
            },
        ]
    }
}


def coops(monkeypatch):
    source = CoopsWaterSource(CoopsProvider())
    monkeypatch.setattr(source.provider, "all_stations",
                        lambda: COOPS_STATIONS["stations"])

    def fake_get_json(url, params, provider=None, **_):
        if params.get("product") == "predictions":
            return COOPS_HILO
        if params.get("station") == "9444090":
            return COOPS_WL
        return {"error": {"message": " No data was found. "}}

    monkeypatch.setattr(noaa_mod, "get_json", fake_get_json)
    return source


def query(lat=48.125, lon=-123.44, **options):
    return Query.from_position(lat=lat, lon=lon, radius_km=30, time=WINDOW, **options)


class TestCoopsMirrorsTheCanadianTideShape:
    def test_discovery_filters_the_station_list_spatially(self, monkeypatch):
        source = coops(monkeypatch)
        matches = source.discover(query())
        assert [m.station_id for m in matches] == ["9444090"]

    def test_observations_and_predictions_never_share_a_node(self, monkeypatch):
        source = coops(monkeypatch)
        series = source.fetch(query(), source.discover(query()))
        paths = {s.node_path for s in series}
        assert paths == {
            "in_situ/tides/9444090", "predictions/tides_hilo/9444090"
        }, "the branch layout must match dfo_tides exactly"

    def test_the_datum_is_stated_where_the_erddap_mirror_lost_it(self, monkeypatch):
        source = coops(monkeypatch)
        obs = next(s for s in source.fetch(query(), source.discover(query()))
                   if s.node_path.startswith("in_situ"))
        assert obs.attrs["datum"] == "MLLW"
        var = obs.var_attrs["water_surface_height_above_reference_datum"]
        assert var["vertical_datum"] == "MLLW"

    def test_values_times_and_flags_survive(self, monkeypatch):
        source = coops(monkeypatch)
        obs = next(s for s in source.fetch(query(), source.discover(query()))
                   if s.node_path.startswith("in_situ"))
        frame = obs.frame
        assert frame.index[0] == pd.Timestamp("2024-07-01T00:00:00Z")
        assert frame["water_surface_height_above_reference_datum"].iloc[0] == 1.241
        assert frame["water_surface_height_above_reference_datum_qc"].iloc[0] == "v"
        # An empty-string value is a gap, not a zero.
        assert pd.isna(frame["water_surface_height_above_reference_datum"].iloc[2])

    def test_a_great_lakes_station_defaults_to_igld(self, monkeypatch):
        source = coops(monkeypatch)
        match = source.discover(query(lat=42.7733, lon=-86.2128))[0]
        assert source._datum(query(lat=42.7733, lon=-86.2128), match) == "IGLD"

    def test_coops_datum_option_wins(self, monkeypatch):
        source = coops(monkeypatch)
        q = query(coops_datum="msl")
        assert source._datum(q, source.discover(q)[0]) == "MSL"

    def test_no_data_found_is_an_answer_not_a_failure(self, monkeypatch):
        source = coops(monkeypatch)

        def all_empty(url, params, provider=None, **_):
            return {"error": {"message": "No data was found."}}

        monkeypatch.setattr(noaa_mod, "get_json", all_empty)
        series = source.fetch(query(), source.discover(query()))
        assert all(s.is_empty for s in series)


class TestNwisTranslation:
    def test_rdb_parsing_skips_comments_and_the_dtype_row(self):
        rows = _parse_rdb(NWIS_RDB)
        assert len(rows) == 5
        assert rows[0]["site_no"] == "12045500" and rows[0]["parm_cd"] == "00060"

    def source(self, monkeypatch):
        source = UsgsWaterSource(UsgsProvider())
        monkeypatch.setattr(usgs_mod, "get_text",
                            lambda url, params, provider=None, **_: NWIS_RDB)
        monkeypatch.setattr(usgs_mod, "get_json",
                            lambda url, params, provider=None, **_: NWIS_IV)
        return source

    def test_a_discontinued_gauge_excludes_itself_by_its_own_dates(self, monkeypatch):
        source = self.source(monkeypatch)
        matches = source.discover(query(lat=48.05, lon=-123.58))
        assert [m.station_id for m in matches] == ["12045500"], (
            "LONG GONE CREEK ended in 1972 and must not be discovered for a 2024 window"
        )
        assert matches[0].first == pd.Timestamp("1897-10-01", tz="UTC")

    def test_the_no_data_sentinel_is_a_gap_and_qualifiers_are_carried(self, monkeypatch):
        source = self.source(monkeypatch)
        q = query(lat=48.05, lon=-123.58)
        (series,) = source.fetch(q, source.discover(q))
        frame = series.frame
        # Timestamps arrive with a local UTC offset and land as UTC.
        assert frame.index[0] == pd.Timestamp("2024-07-01T00:00:00Z")
        assert frame["river_discharge"].iloc[0] == 1000.0
        assert pd.isna(frame["river_discharge"].iloc[1]), "-999999 is a gap, not a flow"
        assert frame["river_discharge_qc"].tolist() == ["A", "P"]
        assert series.var_attrs["river_discharge"]["units"] == "ft3 s-1"

    def test_to_cf_units_reaches_cubic_metres(self, monkeypatch):
        source = self.source(monkeypatch)
        q = query(lat=48.05, lon=-123.58, to_cf_units=True)
        (series,) = source.fetch(q, source.discover(q))
        assert series.frame["river_discharge"].iloc[0] == pytest.approx(28.316846592)
        assert series.var_attrs["river_discharge"]["units"] == "m3 s-1"

    def test_a_grab_sample_only_site_is_not_promised_by_the_iv_source(self, monkeypatch):
        """seriesCatalogOutput lists water-quality grab samples too — 573 'qw' rows in one
        Olympic Peninsula box — and the IV service answers nothing for those sites."""
        source = self.source(monkeypatch)
        ids = [m.station_id for m in source.discover(query(lat=48.05, lon=-123.58))]
        assert "88888888" not in ids

    def test_the_daily_source_serves_means_labelled_by_local_date(self, monkeypatch):
        from omnisea.providers.usgs import UsgsProvider, UsgsWaterDailySource

        source = UsgsWaterDailySource(UsgsProvider())
        monkeypatch.setattr(usgs_mod, "get_text",
                            lambda url, params, provider=None, **_: NWIS_RDB)

        def dv(url, params, provider=None, **_):
            assert url.endswith("/dv/"), "the daily source must ask the DV service"
            assert params["statCd"] == "00003,00001,00002", (
                "mean, max and min in one request; each declares its statistic"
            )
            return {
                "value": {"timeSeries": [{
                    "variable": {"variableCode": [{"value": "00060"}],
                                 "unit": {"unitCode": "ft3/s"}, "noDataValue": -999999.0,
                                 "options": {"option": [{"name": "Statistic",
                                                         "optionCode": "00003"}]}},
                    "values": [{"value": [
                        {"value": "1020", "qualifiers": ["A"],
                         "dateTime": "2024-07-01T00:00:00.000"},
                    ]}],
                }]}
            }

        monkeypatch.setattr(usgs_mod, "get_json", dv)
        q = query(lat=48.05, lon=-123.58)
        (series,) = source.fetch(q, source.discover(q))
        assert series.node_path == "in_situ/hydrometric_daily/12045500"
        assert series.frame["river_discharge"].iloc[0] == 1020.0
        assert series.var_attrs["river_discharge"]["cell_methods"] == "time: mean"
        assert "LOCAL_DATE" in series.attrs["time_reference"]
        assert series.attrs["omnisea_period"] == "D"

    def test_us_and_canadian_rivers_share_a_branch(self, monkeypatch):
        source = self.source(monkeypatch)
        q = query(lat=48.05, lon=-123.58)
        (series,) = source.fetch(q, source.discover(q))
        assert series.node_path == "in_situ/hydrometric/12045500"
        tree = build_tree(Query.from_sites([Site(48.05, -123.58, "Elwha")], WINDOW), [series])
        assert "/in_situ/hydrometric/12045500" in {n.path for n in tree.subtree}
        assert "usgs" in omnisea.providers() and "noaa_coops" in omnisea.providers()

    def test_daily_max_and_min_carry_their_own_cell_methods(self, monkeypatch):
        from omnisea.providers.usgs import UsgsProvider, UsgsWaterDailySource

        source = UsgsWaterDailySource(UsgsProvider())
        monkeypatch.setattr(usgs_mod, "get_text",
                            lambda url, params, provider=None, **_: NWIS_RDB)

        def dv(url, params, provider=None, **_):
            def series(stat, value):
                return {
                    "variable": {"variableCode": [{"value": "00060"}],
                                 "unit": {"unitCode": "ft3/s"}, "noDataValue": -999999.0,
                                 "options": {"option": [{"name": "Statistic",
                                                         "optionCode": stat}]}},
                    "values": [{"value": [{"value": value, "qualifiers": ["A"],
                                           "dateTime": "2024-07-01T00:00:00.000"}]}],
                }
            return {"value": {"timeSeries": [
                series("00003", "1020"), series("00001", "1310"), series("00002", "890"),
            ]}}

        monkeypatch.setattr(usgs_mod, "get_json", dv)
        q = query(lat=48.05, lon=-123.58)
        (result,) = source.fetch(q, source.discover(q))
        frame = result.frame
        assert frame["river_discharge"].iloc[0] == 1020.0
        assert frame["river_discharge_max"].iloc[0] == 1310.0
        assert frame["river_discharge_min"].iloc[0] == 890.0
        assert result.var_attrs["river_discharge"]["cell_methods"] == "time: mean"
        assert result.var_attrs["river_discharge_max"]["cell_methods"] == "time: maximum"
        assert result.var_attrs["river_discharge_min"]["cell_methods"] == "time: minimum"

    def test_an_uncurated_parameter_code_passes_through_marked_unmapped(self, monkeypatch):
        source = self.source(monkeypatch)

        def iv(url, params, provider=None, **_):
            assert "00095" in params["parameterCd"]
            return {"value": {"timeSeries": [{
                "variable": {"variableCode": [{"value": "00095"}],
                             "variableName": "Specific conductance",
                             "unit": {"unitCode": "uS/cm @25C"}, "noDataValue": -999999.0},
                "values": [{"value": [{"value": "212", "qualifiers": ["A"],
                                       "dateTime": "2024-07-01T00:00:00.000-07:00"}]}],
            }]}}

        monkeypatch.setattr(usgs_mod, "get_json", iv)
        q = query(lat=48.05, lon=-123.58, usgs_parameters=["00095"])
        (result,) = source.fetch(q, source.discover(q))
        assert result.frame["nwis_00095"].iloc[0] == 212.0
        attrs = result.var_attrs["nwis_00095"]
        assert attrs["omnisea_mapped"] == 0
        assert attrs["units"] == "uS/cm @25C"
        assert "standard_name" not in attrs


class TestCoopsEras:
    def test_a_pre_1996_window_plans_hourly_height_requests(self, monkeypatch):
        source = coops(monkeypatch)
        q = Query.from_position(lat=48.125, lon=-123.44, radius_km=30,
                                time=("1985-06-01", "1996-02-01"))
        plan = list(source._requests(q, "water_level"))
        products = {p for p, _, _ in plan}
        assert products == {"hourly_height", "water_level"}, (
            "the archive era and the six-minute era split at 1996; before this, a pre-1996 "
            "request simply failed and the caller never learned why"
        )
        archive_end = max(e for p, _, e in plan if p == "hourly_height")
        modern_start = min(s for p, s, _ in plan if p == "water_level")
        assert archive_end == modern_start == pd.Timestamp("1996-01-01", tz="UTC")

    def test_a_modern_window_stays_six_minute_only(self, monkeypatch):
        source = coops(monkeypatch)
        plan = list(source._requests(query(), "water_level"))
        assert {p for p, _, _ in plan} == {"water_level"}

    def test_observed_extrema_are_off_by_default_and_land_under_in_situ(self, monkeypatch):
        source = coops(monkeypatch)
        series = source.fetch(query(), source.discover(query()))
        assert not any("tides_extrema" in s.node_path for s in series)
        q = query(coops_high_low=True)
        series = source.fetch(q, source.discover(q))
        extrema = [s for s in series if "tides_extrema" in s.node_path]
        assert extrema and extrema[0].node_path.startswith("in_situ/"), (
            "high_low is a measurement product despite its prediction-flavoured name"
        )

    def test_the_datums_ladder_rides_on_the_observation_node(self, monkeypatch):
        source = coops(monkeypatch)
        monkeypatch.setattr(
            source.provider, "datum_ladder",
            lambda sid: {"datum_epoch": "1983-2001", "orthometric_datum": "NAVD88",
                         "datum_offset_MHHW": 2.383, "datum_offsets_units": "m"},
        )
        obs = next(s for s in source.fetch(query(), source.discover(query()))
                   if s.node_path.startswith("in_situ/tides/"))
        assert obs.attrs["datum_epoch"] == "1983-2001"
        assert obs.attrs["datum_offset_MHHW"] == 2.383
