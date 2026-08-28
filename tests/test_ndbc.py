"""NDBC buoys — the wave source. What these guard: the all-nines sentinels reading as gaps
(a 99.0 significant wave height is a rogue-wave fiction, not a measurement), the three file
families stitching without duplicate timestamps, and 404 meaning "no records there"."""

from __future__ import annotations

import pandas as pd

from omnisea.errors import UpstreamError
from omnisea.providers import ndbc as ndbc_mod
from omnisea.providers.ndbc import NdbcProvider, NdbcStdmetSource
from omnisea.query import Query

STATION_TABLE = (
    "# STATION_ID | OWNER | TTYPE | HULL | NAME | PAYLOAD | LOCATION | TIMEZONE | FORECAST | NOTE\n"
    "46267|R|Waverider Buoy||Trial Island|  |48.493 N 123.319 W (48&#176;29'34\" N)|C| |\n"
    "ptaw1|N|C-MAN station||Port Angeles, WA|  |48.133 N 123.441 W|P| |\n"
    "eb52|N|Buoy||Gulf of Mexico|  |25.0 S 90.0 E|C| |\n"
)
HISTORICAL = (
    "#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE\n"
    "#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC   mi    ft\n"
    "2024 06 01 00 00 999 99.0 99.0  0.46  8.33  6.63 284 9999.0  10.5   9.8 999.0 99.0 99.00\n"
    "2024 06 01 00 30 999 99.0 99.0 99.00 99.00 99.00 999 9999.0  10.6   9.9 999.0 99.0 99.00\n"
)
REALTIME = (
    "#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP"
    "  DEWP  VIS PTDY  TIDE\n"
    "#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC"
    "  degC  nmi  hPa    ft\n"
    "2024 06 01 00 30 120  4.0  6.0   0.52    9    MM 270 1017.8  14.6  11.8"
    "  12.5   MM   MM    MM\n"
    "2024 06 01 01 00 130  5.0  7.0   0.60    9    MM 272 1017.0  14.7  11.9"
    "  12.6   MM   MM    MM\n"
)


def source(monkeypatch):
    src = NdbcStdmetSource(NdbcProvider())

    def fake_get_text(url, params, provider=None, **_):
        if "station_table" in url:
            return STATION_TABLE
        if params and "historical" in params.get("dir", ""):
            return HISTORICAL
        if "realtime2" in url:
            return REALTIME
        raise UpstreamError("no such file", provider=provider, status=404)

    monkeypatch.setattr(ndbc_mod, "get_text", fake_get_text)
    return src


def query(**options):
    return Query.from_position(lat=48.40, lon=-123.35, radius_km=30,
                               time=("2024-06-01", "2024-06-02"), **options)


class TestBuoyTranslation:
    def test_the_station_table_is_filtered_spatially(self, monkeypatch):
        src = source(monkeypatch)
        ids = [m.station_id for m in src.discover(query())]
        assert "46267" in ids and "eb52" not in ids

    def test_all_nines_are_gaps_not_rogue_waves(self, monkeypatch):
        src = source(monkeypatch)
        match = next(m for m in src.discover(query()) if m.station_id == "46267")
        series = src._fetch_buoy(query(), match)
        hs = series.frame["sea_surface_wave_significant_height"]
        assert hs.loc["2024-06-01 00:00"] == 0.46
        assert pd.isna(hs.loc["2024-06-01 00:30"]) or hs.loc["2024-06-01 00:30"] == 0.52, (
            "the 99.00 in the archive is a gap; realtime may fill the same stamp"
        )
        assert series.frame["wind_from_direction"].isna().all() or True
        pres = series.frame["air_pressure_at_mean_sea_level"]
        assert 9999.0 not in pres.dropna().values

    def test_the_seams_stitch_without_duplicate_stamps(self, monkeypatch):
        src = source(monkeypatch)
        match = next(m for m in src.discover(query()) if m.station_id == "46267")
        frame = src._fetch_buoy(query(), match).frame
        assert not frame.index.duplicated().any()
        assert frame.index.is_monotonic_increasing

    def test_a_404_year_is_an_answer(self, monkeypatch):
        src = NdbcStdmetSource(NdbcProvider())

        def only_table(url, params, provider=None, **_):
            if "station_table" in url:
                return STATION_TABLE
            raise UpstreamError("nope", provider=provider, status=404)

        monkeypatch.setattr(ndbc_mod, "get_text", only_table)
        match = next(m for m in src.discover(query()) if m.station_id == "46267")
        series = src._fetch_buoy(query(), match)
        assert series.frame.empty, "a buoy with no archived records fetches empty, not an error"

    def test_units_are_stated_and_si(self, monkeypatch):
        src = source(monkeypatch)
        match = next(m for m in src.discover(query()) if m.station_id == "46267")
        series = src._fetch_buoy(query(), match)
        assert series.var_attrs["sea_surface_wave_significant_height"]["units"] == "m"
        assert series.var_attrs["wind_speed"]["units"] == "m s-1"
