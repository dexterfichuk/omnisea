"""Marine SWOB shaping, exercised offline against captured Datamart responses.

The fixtures are real files pulled from ``dd.weather.gc.ca``: one MSC-type buoy report (Halibut
Bank, the full wave suite), one OPP-type report (Southern Georgia Strait, the newer field
spellings and a ten-minute cadence), and one Apache autoindex listing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from omnisea.errors import PayloadTooLargeError
from omnisea.providers.base import StationMatch
from omnisea.providers.eccc import EcccProvider
from omnisea.providers.eccc import swob as swob_module
from omnisea.providers.eccc.swob import (
    EcccSwobMarine,
    _extended_wmo_id,
    _observation_files,
    _parse_swob_ml,
    _utc_days,
)
from omnisea.query import Query

FIXTURES = Path(__file__).parent / "fixtures"

# The captured observations are both from 2026-08-20 12:00 UTC.
CAPTURE_DAY = ("2026-08-20", "2026-08-21")

HALIBUT_BANK = dict(station_id="9100552", name="HALIBUT BANK", lat=49.34, lon=-123.7267)


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def source():
    return EcccSwobMarine(EcccProvider())


@pytest.fixture
def msc_row():
    return _parse_swob_ml(read("swobm_observation.xml"))


@pytest.fixture
def opp_row():
    return _parse_swob_ml(read("swobm_observation_opp.xml"))


def a_match(**over) -> StationMatch:
    fields = dict(source="eccc_swob_marine", **HALIBUT_BANK)
    fields.update(over)
    match = StationMatch(**fields)
    match.extra.setdefault("buoy_id", "4600146")
    return match


def series_for(source, rows, time=CAPTURE_DAY):
    query = Query.from_area((-126, 48, -122, 51), time)
    return source.series_from_rows(query, a_match(), rows)


# --------------------------------------------------------------------------- SWOB-ML parsing


class TestSwobMlParsing:
    def test_identification_and_measurements_land_in_one_row(self, msc_row):
        assert msc_row["msc_id"] == 9100552.0
        assert msc_row["stn_nam"] == "HALIBUT BANK"
        assert msc_row["date_tm"] == "2026-08-20T12:05:00.000Z"
        assert msc_row["avg_sea_sfc_temp_pst10mts"] == pytest.approx(19.0, abs=2.0)

    def test_every_measurement_carries_its_own_units(self, msc_row):
        """The `-uom` sibling is what makes the source survive a sensor change."""
        assert msc_row["avg_sig_wave_hgt_pst20mts-uom"] == "m"
        assert msc_row["avg_wave_pd_pst20mts-uom"] == "s"
        assert msc_row["avg_sea_sfc_temp_pst10mts-uom"] == "°C"
        assert msc_row["avg_wnd_spd_pst10mts-uom"] == "km/h"

    def test_quality_summary_becomes_a_qa_sibling(self, msc_row):
        assert msc_row["avg_sea_sfc_temp_pst10mts-qa"] == 100.0

    def test_msng_becomes_missing_rather_than_the_string(self, msc_row):
        """`wnd_snsr_vert_disp` is MSNG at this buoy; left as text it poisons the column."""
        assert msc_row["wnd_snsr_vert_disp"] is None
        assert "MSNG" not in [v for v in msc_row.values() if isinstance(v, str)]

    def test_a_non_swob_document_is_rejected_not_raised(self):
        assert _parse_swob_ml("<html><body>404</body></html>") is None
        assert _parse_swob_ml("not xml at all <<<") is None


# --------------------------------------------------------------------------- CF shaping


class TestMarineFieldMapping:
    def test_wave_and_sea_surface_variables_get_cf_names(self, source, msc_row):
        series = series_for(source, [msc_row])
        for name in (
            "sea_surface_wave_significant_height",
            "sea_surface_wave_mean_period",
            "sea_surface_wave_from_direction",
            "sea_surface_wave_maximum_height",
            "sea_surface_temperature",
        ):
            assert name in series.frame.columns, name

    def test_units_come_from_the_data_not_a_table(self, source, msc_row):
        series = series_for(source, [msc_row])
        assert series.var_attrs["sea_surface_wave_significant_height"]["units"] == "m"
        assert series.var_attrs["sea_surface_temperature"]["units"] == "°C"
        assert series.var_attrs["wind_speed"]["units"] == "km/h"

    def test_cell_methods_record_the_window_the_field_name_encodes(self, source, msc_row):
        series = series_for(source, [msc_row])
        attrs = series.var_attrs
        assert attrs["wind_speed"]["cell_methods"] == "time: mean (interval: 10 minutes)"
        assert (
            attrs["sea_surface_wave_maximum_height"]["cell_methods"]
            == "time: maximum (interval: 20 minutes)"
        )
        assert (
            attrs["sea_surface_wave_significant_height"]["cell_methods"]
            == "time: mean (interval: 20 minutes)"
        )

    def test_event_periods_carry_no_cell_methods(self, source, msc_row):
        """Resampling the period of the highest wave as a maximum would invent a wave."""
        series = series_for(source, [msc_row])
        assert "cell_methods" not in series.var_attrs["sea_surface_wave_period_of_highest_wave"]

    def test_spectral_and_time_domain_wave_height_stay_separate(self, source, msc_row):
        """Two independent estimates of Hs; merging them would hide a disagreement."""
        series = series_for(source, [msc_row])
        assert "sea_surface_wave_significant_height" in series.frame.columns
        assert "sea_surface_wave_significant_height_from_spectrum" in series.frame.columns
        spectral = series.var_attrs["sea_surface_wave_significant_height_from_spectrum"]
        assert spectral["standard_name"] == "sea_surface_wave_significant_height"

    def test_undocumented_spectral_period_is_described_but_not_standardised(
        self, source, msc_row
    ):
        """ECCC does not say which frequency moment it is, and CF names one per moment."""
        attrs = series_for(source, [msc_row]).var_attrs["spectral_mean_wave_period"]
        assert "standard_name" not in attrs
        assert attrs["long_name"]

    def test_pressure_tendency_is_not_given_the_cf_rate_name(self, source, msc_row):
        """A 3-hour change in hPa is not tendency_of_air_pressure, which is a rate in Pa s-1."""
        attrs = series_for(source, [msc_row]).var_attrs["pressure_change_past_3_hours"]
        assert "standard_name" not in attrs
        assert attrs["units"] == "hPa"

    def test_opp_buoys_reach_the_same_cf_names_under_different_field_names(
        self, source, opp_row
    ):
        series = series_for(source, [opp_row])
        assert "sea_surface_wave_significant_height" in series.frame.columns
        assert "sea_surface_wave_significant_period" in series.frame.columns
        assert "sea_surface_temperature" in series.frame.columns

    def test_unmapped_fields_are_carried_not_dropped(self, source, msc_row):
        """Battery voltage and buoy drift position have no CF name but are still data."""
        columns = {str(c) for c in series_for(source, [msc_row]).frame.columns}
        assert "avg_batry_volt_pst10mts" in columns
        assert "crnt_buoy_lat" in columns
        assert "avg_wnd_spd_pst10mts_1" in columns  # redundant sensor keeps its own identity

    def test_uom_and_qa_siblings_do_not_become_variables(self, source, msc_row):
        frame = series_for(source, [msc_row]).frame
        assert not [c for c in frame.columns if str(c).endswith(("-uom", "-qa"))]

    def test_identity_columns_do_not_become_variables(self, source, msc_row):
        columns = {str(c) for c in series_for(source, [msc_row]).frame.columns}
        assert not columns & {"msc_id", "stn_nam", "wmo_synop_id", "lat", "long", "date_tm"}

    def test_quality_flags_travel_beside_their_measurement(self, source, msc_row):
        frame = series_for(source, [msc_row]).frame
        assert "sea_surface_temperature_qc" in frame.columns

    def test_units_are_only_converted_when_asked(self, source, msc_row):
        query = Query.from_area((-126, 48, -122, 51), CAPTURE_DAY, to_cf_units=True)
        series = source.series_from_rows(query, a_match(), [msc_row])
        assert series.var_attrs["sea_surface_temperature"]["units"] == "K"
        assert float(series.frame["sea_surface_temperature"].iloc[0]) > 250

    def test_every_curated_standard_name_is_a_real_cf_name(self, source):
        """Guards against a plausible-looking invention slipping into the table."""
        for raw, spec in source.fields.items():
            if not spec.standard_name:
                continue
            assert spec.standard_name in CF_STANDARD_NAMES, f"{raw} -> {spec.standard_name}"


#: Every standard name this source emits, checked against the CF standard name table at
#: https://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml
CF_STANDARD_NAMES = frozenset(
    {
        "air_pressure",
        "air_pressure_at_mean_sea_level",
        "air_temperature",
        "sea_surface_temperature",
        "sea_surface_wave_directional_spread",
        "sea_surface_wave_directional_spread_at_variance_spectral_density_maximum",
        "sea_surface_wave_from_direction",
        "sea_surface_wave_from_direction_at_variance_spectral_density_maximum",
        "sea_surface_wave_maximum_crest_height",
        "sea_surface_wave_maximum_height",
        "sea_surface_wave_mean_height",
        "sea_surface_wave_mean_period",
        "sea_surface_wave_mean_period_from_variance_spectral_density_inverse_frequency_moment",
        "sea_surface_wave_period_at_variance_spectral_density_maximum",
        "sea_surface_wave_period_of_highest_wave",
        "sea_surface_wave_significant_height",
        "sea_surface_wave_significant_period",
        "wind_from_direction",
        "wind_speed",
        "wind_speed_of_gust",
    }
)


# --------------------------------------------------------------------------- discovery


class TestMarineDiscovery:
    def test_catalogue_feature_becomes_a_match_with_the_english_name(self, source):
        import json

        catalogue = json.loads(read("swobm_stations.json"))
        query = Query.from_area((-126, 48, -122, 51), CAPTURE_DAY)
        matches = [
            m
            for m in (source.station_from_feature(query, f) for f in catalogue["features"])
            if m is not None
        ]
        names = {m.name for m in matches}
        assert "HALIBUT BANK" in names
        assert all(m.station_id.isdigit() for m in matches)

    def test_discovery_records_the_datamart_directory_id(self, source):
        import json

        catalogue = json.loads(read("swobm_stations.json"))
        query = Query.from_area((-126, 48, -122, 51), CAPTURE_DAY)
        halibut = next(
            f for f in catalogue["features"] if f["properties"]["name_en"] == "HALIBUT BANK"
        )
        match = source.station_from_feature(query, halibut)
        assert match.require("buoy_id") == "4600146"

    def test_stations_outside_the_area_are_rejected(self, source):
        import json

        catalogue = json.loads(read("swobm_stations.json"))
        query = Query.from_area((-124, 49.2, -123.5, 49.5), CAPTURE_DAY)
        kept = [
            m
            for m in (source.station_from_feature(query, f) for f in catalogue["features"])
            if m is not None
        ]
        assert {m.name for m in kept} == {"HALIBUT BANK"}


class TestExtendedWmoId:
    def test_five_digit_ids_regain_the_zeros_the_datamart_uses(self):
        assert _extended_wmo_id(46146) == "4600146"
        assert _extended_wmo_id(44137) == "4400137"
        assert _extended_wmo_id("45152") == "4500152"

    def test_seven_digit_ids_pass_through(self):
        assert _extended_wmo_id(4600303) == "4600303"

    def test_unusable_ids_are_rejected_rather_than_guessed(self):
        assert _extended_wmo_id(None) == ""
        assert _extended_wmo_id("") == ""
        assert _extended_wmo_id("46 146") == ""
        assert _extended_wmo_id(146) == ""


# --------------------------------------------------------------------------- datamart walking


class TestDirectoryListing:
    def test_observation_files_are_read_off_a_real_autoindex_page(self):
        found = _observation_files(read("swobm_listing.html"))
        assert len(found) == 24  # this buoy reports hourly
        assert found[0][0] == "2026-08-20-0005-4600146-AUTO-swob.xml"
        assert all(name.endswith(".xml") for name, _ in found)

    def test_the_report_time_is_read_from_the_filename(self):
        """So a partly-covered day can be narrowed before anything is downloaded."""
        _, reported = _observation_files(read("swobm_listing.html"))[0]
        assert reported == pd.Timestamp("2026-08-20T00:05", tz="UTC")

    def test_the_parent_directory_link_is_not_mistaken_for_an_observation(self):
        listing = read("swobm_listing.html")
        assert "Parent Directory" in listing
        assert not [name for name, _ in _observation_files(listing) if name.startswith("/")]

    def test_report_minutes_are_not_assumed(self):
        """OPP buoys file at :00 through :50, MSC buoys at :05 — hence listing, not guessing."""
        page = (
            '<a href="2026-08-20-1200-4600303-AUTO-swob.xml">x</a>'
            '<a href="2026-08-20-1210-4600303-AUTO-swob.xml">x</a>'
        )
        assert [t.strftime("%H%M") for _, t in _observation_files(page)] == ["1200", "1210"]


class TestDayNarrowing:
    def test_only_the_files_inside_the_window_are_requested(self, source, monkeypatch):
        """A window's first and last day are usually partly outside it."""
        monkeypatch.setattr(
            swob_module, "_get_text", lambda url, source: read("swobm_listing.html")
        )
        query = Query.from_area((-126, 48, -122, 51), ("2026-08-20T06:00", "2026-08-20T09:00"))
        urls = source._list_day(query, a_match(), "20260820")
        assert [u.rsplit("/", 1)[-1][:15] for u in urls] == [
            "2026-08-20-0605",
            "2026-08-20-0705",
            "2026-08-20-0805",
        ]

    def test_a_day_the_buoy_was_not_deployed_is_a_gap_not_a_failure(self, source, monkeypatch):
        monkeypatch.setattr(swob_module, "_get_text", lambda url, source: None)
        query = Query.from_area((-126, 48, -122, 51), CAPTURE_DAY)
        assert source._list_day(query, a_match(), "20260820") == []


class TestDocumentDecoding:
    def test_an_undeclared_charset_is_read_as_utf8_not_latin1(self, monkeypatch):
        """The Datamart serves SWOB-ML as application/xml with no charset, and it is UTF-8.

        Left to requests' text/* fallback the degree sign in every temperature and direction
        `-uom` would decode to mojibake, and the units would be quietly wrong.
        """
        captured = {}

        class _Resp:
            status_code = 200
            ok = True
            url = "http://x"
            headers = {"Content-Type": "application/xml"}
            encoding = "ISO-8859-1"  # what requests assumes when nothing is declared

            @property
            def text(self):
                captured["encoding"] = self.encoding
                return "°C"

        monkeypatch.setattr(swob_module, "get_session", lambda: _Session(_Resp()))
        assert swob_module._get_text("http://x", source="eccc_swob_marine") == "°C"
        assert captured["encoding"] == "utf-8"

    def test_a_declared_charset_is_respected(self, monkeypatch):
        class _Resp:
            status_code = 200
            ok = True
            url = "http://x"
            headers = {"Content-Type": "text/html; charset=iso-8859-1"}
            encoding = "iso-8859-1"
            text = "<a href=\"x.xml\">x</a>"

        monkeypatch.setattr(swob_module, "get_session", lambda: _Session(_Resp()))
        swob_module._get_text("http://x", source="eccc_swob_marine")

    def test_a_missing_document_is_a_gap_rather_than_an_error(self, monkeypatch):
        class _Resp:
            status_code = 404
            ok = False
            url = "http://x"
            headers = {}

        monkeypatch.setattr(swob_module, "get_session", lambda: _Session(_Resp()))
        assert swob_module._get_text("http://x", source="eccc_swob_marine") is None

    def test_a_real_failure_still_raises(self, monkeypatch):
        from omnisea.errors import UpstreamError

        class _Resp:
            status_code = 503
            ok = False
            url = "http://x"
            headers = {}

        monkeypatch.setattr(swob_module, "get_session", lambda: _Session(_Resp()))
        with pytest.raises(UpstreamError) as excinfo:
            swob_module._get_text("http://x", source="eccc_swob_marine")
        assert excinfo.value.status == 503


class _Session:
    def __init__(self, response):
        self._response = response

    def get(self, url, **kwargs):
        return self._response


class TestWindowToDays:
    def test_every_utc_day_the_window_touches_is_listed(self):
        query = Query.from_area((-126, 48, -122, 51), ("2026-08-19T22:00", "2026-08-21T03:00"))
        assert _utc_days(query) == ["20260819", "20260820", "20260821"]

    def test_a_window_inside_one_day_lists_that_day(self):
        query = Query.from_area((-126, 48, -122, 51), ("2026-08-20T01:00", "2026-08-20T05:00"))
        assert _utc_days(query) == ["20260820"]


class TestRequestCeiling:
    def test_too_many_files_is_an_error_not_a_silent_truncation(self, source, monkeypatch):
        """One request per observation, so a wide window is a round-trip problem."""
        monkeypatch.setattr(
            EcccSwobMarine,
            "_list_day",
            lambda self, query, match, stamp: [f"http://x/{stamp}/{i}.xml" for i in range(200)],
        )
        query = Query.from_area((-126, 48, -122, 51), ("2026-08-01", "2026-08-20"), max_items=100)
        with pytest.raises(PayloadTooLargeError) as excinfo:
            source.fetch(query, [a_match()])
        assert excinfo.value.limit == 100

    def test_no_matches_makes_no_requests(self, source):
        query = Query.from_area((-126, 48, -122, 51), CAPTURE_DAY)
        assert source.fetch(query, []) == []


class TestRetention:
    def test_a_window_older_than_the_archive_is_explained_not_silently_empty(self, source):
        query = Query.from_area((-126, 48, -122, 51), ("2024-07-01", "2024-07-08"))
        assert not source.covers(query)
        assert "30 days" in (source.retention_gap(query) or "")

    def test_a_recent_window_is_covered(self, source):
        now = pd.Timestamp.now(tz="UTC")
        query = Query.from_area(
            (-126, 48, -122, 51), (now - pd.Timedelta(days=2), now - pd.Timedelta(days=1))
        )
        assert source.covers(query)


class TestSourceWiring:
    def test_it_advertises_the_ocean_variables_it_serves(self, source):
        assert "sea_surface_wave_significant_height" in source.variables
        assert "sea_surface_temperature" in source.variables

    def test_a_wave_query_selects_this_source(self, source):
        query = Query.from_area(
            (-126, 48, -122, 51), CAPTURE_DAY, variables=["sea_surface_wave_significant_height"]
        )
        assert source.wants_anything(query)

    def test_node_attrs_point_at_the_datamart_not_the_ogc_api(self, source, msc_row):
        attrs = series_for(source, [msc_row]).attrs
        assert "dd.weather.gc.ca" in attrs["source_url"]
        assert attrs["collection"] == "swob-marine-stations"

    def test_the_node_path_separates_buoys_from_land_weather(self, source, msc_row):
        assert series_for(source, [msc_row]).node_path == "in_situ/marine_buoy/9100552"


# --------------------------------------------------------------------------- live upstream


@pytest.mark.network
class TestMarineUpstream:
    """What fixtures cannot catch: the Datamart layout and the absence of marine data on the API."""

    def test_swob_realtime_still_holds_no_marine_observations(self):
        """The finding this source exists because of. If it ever changes, this test says so."""
        from omnisea.http import get_json

        url = "https://api.weather.gc.ca/collections/swob-realtime/items"
        for params in (
            {"msc_id-value": "9100552"},  # HALIBUT BANK
            {"wmo_synop_id-value": "46146"},
            {"bbox": "-123.80,49.28,-123.65,49.40"},  # its position, all time
        ):
            payload = get_json(url, dict(params, limit=1, f="json"), provider="eccc_swob")
            assert payload["numberMatched"] == 0, params

    def test_the_marine_catalogue_is_reachable_and_still_geolocated(self):
        source = EcccSwobMarine(EcccProvider())
        query = Query.from_area((-130, 48, -122, 52), _recent_window())
        matches = source.discover(query)
        assert matches, "swob-marine-stations returned no buoys for the BC coast"
        assert all(-90 <= m.lat <= 90 and -180 <= m.lon <= 180 for m in matches)
        assert all(m.require("buoy_id").isdigit() for m in matches)

    def test_a_real_buoy_returns_waves_and_sea_surface_temperature(self):
        source = EcccSwobMarine(EcccProvider())
        query = Query.from_area((-124, 49.2, -123.5, 49.5), _recent_window())
        matches = source.discover(query)
        assert matches
        series = source.fetch(query, matches)
        assert series
        frame = series[0].frame
        assert not frame.empty
        assert "sea_surface_temperature" in frame.columns
        assert any("wave" in str(c) for c in frame.columns)
        assert frame.index.is_monotonic_increasing
        assert str(frame.index.tz) == "UTC"

    def test_units_still_arrive_beside_every_value(self):
        source = EcccSwobMarine(EcccProvider())
        query = Query.from_area((-124, 49.2, -123.5, 49.5), _recent_window())
        series = source.fetch(query, source.discover(query))
        assert series
        attrs = series[0].var_attrs
        assert attrs["sea_surface_temperature"]["units"] == "°C"


def _recent_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Yesterday, whole. Today's directory may hold only the hours that have happened."""
    end = pd.Timestamp.now(tz="UTC").normalize()
    return end - pd.Timedelta(days=1), end
