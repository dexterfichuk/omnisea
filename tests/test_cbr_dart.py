"""Columbia River DART — dams and salmon. What these guard: the year:project:series column
grammar surviving parse, fish counts as day-sums under plain names (there is no CF standard
name for a salmon), local calendar dates handled like every other daily archive, and DART's
habit of answering an unusable request with its HTML homepage instead of an error code."""

from __future__ import annotations

from omnisea.providers import cbr as cbr_mod
from omnisea.providers.cbr import CbrProvider, DartPassageSource, DartRiverSource
from omnisea.query import Query

RIVER_CSV = (
    "\n"
    "mm/dd,2024:BON:outflow (kcfs),2024:BON:spill (kcfs),2024:BON:inflow (kcfs),"
    "2024:BON:disgas (mmHg)\n"
    "7/1,176.700,94.722,168.800,834.944\n"
    "7/2,167.933,94.733,168.200,823.222\n"
    "\n"
    "Notes:\n"
    "Columbia River DART\n"
)
ADULT_CSV = (
    "Project,Date,Chinook Run,Chin,JChin,Stlhd,WStlhd,Sock,Coho,JCoho,Shad,LmpryDay,"
    "LmpryNight,LmpryCombined,LmpryLPS,BTrout,Chum,Pink,TempC\n"
    "Bonneville,2024-07-01,Su,1257,176,678,355,26359,,,49840,328,393,721,1176,,1,,18.6\n"
    "Bonneville,2024-07-02,Su,1211,227,749,341,25554,,,33128,768,824,1592,782,,,,18.7\n"
    "Notes:\n"
    "Columbia River DART\n"
)


def query(**options):
    return Query.from_position(lat=45.644, lon=-121.941, radius_km=5,
                               time=("2024-07-01", "2024-07-02"), **options)


class TestDartTranslation:
    def river(self, monkeypatch, text=RIVER_CSV):
        source = DartRiverSource(CbrProvider())
        monkeypatch.setattr(cbr_mod, "get_text",
                            lambda url, params, provider=None, **_: text)
        return source

    def passage(self, monkeypatch):
        source = DartPassageSource(CbrProvider())
        monkeypatch.setattr(cbr_mod, "get_text",
                            lambda url, params, provider=None, **_: ADULT_CSV)
        return source

    def test_only_the_dam_inside_the_circle_is_discovered(self, monkeypatch):
        source = self.river(monkeypatch)
        assert [m.station_id for m in source.discover(query())] == ["BON"]

    def test_the_column_grammar_parses_and_values_match_the_report(self, monkeypatch):
        source = self.river(monkeypatch)
        (series,) = source.fetch(query(), source.discover(query()))
        frame = series.frame
        assert frame["river_discharge"].iloc[0] == 176.700
        assert frame["spill_discharge"].iloc[0] == 94.722
        assert frame["total_dissolved_gas_pressure"].iloc[0] == 834.944
        assert series.var_attrs["river_discharge"]["cell_methods"] == "time: mean"
        assert "LOCAL_DATE" in series.attrs["time_reference"]

    def test_to_cf_units_scales_the_numbers_with_the_label(self, monkeypatch):
        source = self.river(monkeypatch)
        q = query(to_cf_units=True)
        (series,) = source.fetch(q, source.discover(q))
        assert round(float(series.frame["river_discharge"].iloc[0]), 1) == 5003.6
        assert series.var_attrs["river_discharge"]["units"] == "m3 s-1"

    def test_fish_are_day_sums_under_plain_names(self, monkeypatch):
        source = self.passage(monkeypatch)
        (series,) = source.fetch(query(), source.discover(query()))
        frame = series.frame
        assert frame["sockeye"].iloc[0] == 26359
        assert frame["chinook"].iloc[0] == 1257
        assert frame["chum"].iloc[0] == 1
        assert series.var_attrs["sockeye"]["cell_methods"] == "time: sum"
        assert series.var_attrs["sockeye"].get("standard_name") in (None, ""), (
            "there is no CF standard name for a salmon, and pretending otherwise would be "
            "worse than saying so"
        )
        assert frame["water_temperature"].iloc[0] == 18.6
        assert frame["chinook_run"].iloc[0] == "Su"

    def test_an_html_answer_is_empty_not_a_crash(self, monkeypatch):
        source = self.river(monkeypatch, text="<!DOCTYPE html><html>DART</html>")
        (series,) = source.fetch(query(), source.discover(query()))
        assert series.is_empty

    def test_dams_are_addressable_by_id(self, monkeypatch):
        source = self.river(monkeypatch)
        assert source.locate("bon") == (45.644, -121.941, "Bonneville Dam")
        assert source.locate("NOPE") is None
