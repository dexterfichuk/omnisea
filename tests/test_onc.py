"""Ocean Networks Canada: token handling, the columnar transpose, and citation passthrough.

The offline tests run against captured real responses from Oceans 3.0 (Folger Passage, the ONC
node in Barkley Sound — the same water as the library's running Bamfield example). The token
was scrubbed out of every fixture before it was written.

Live tests are skipped unless ``ONC_TOKEN`` is set, because ONC is the one built-in source that
cannot be reached without a credential.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest
from conftest import FIXTURES

import omnisea
from omnisea.errors import ProviderError, QueryError, UpstreamError
from omnisea.http import redact_url
from omnisea.providers.base import StationMatch
from omnisea.providers.onc import (
    PROPERTY_FIELDS,
    OncProvider,
    OncScalarDataSource,
    _means_no_data,
    clear_cache,
)
from omnisea.query import Query

BAMFIELD = dict(lat=48.8353, lon=-125.1358)
FOLGER = dict(lat=48.8145, lon=-125.2825)
DAY = ("2024-07-01", "2024-07-02")
#: The window tests/fixtures/onc_scalardata.json was actually captured over.
FIXTURE_WINDOW = ("2024-07-01T12:00:00Z", "2024-07-01T16:00:00Z")
TOKEN = "test-token-not-a-real-one"

live = pytest.mark.network
needs_token = pytest.mark.skipif(
    not os.environ.get("ONC_TOKEN"),
    reason="ONC requires a credential; set ONC_TOKEN to run this",
)


@pytest.fixture(autouse=True)
def _clean():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def provider():
    return OncProvider()


@pytest.fixture
def source(provider):
    return OncScalarDataSource(provider)


def load(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def locations():
    """74 real ONC locations within 40 km of Bamfield."""
    return load("onc_locations.json")


@pytest.fixture
def scalardata():
    """A real 4-hour CTD response from Folger Pinnacle, resampled to 10 minutes.

    Chosen because its full eight-sensor suite was reporting: it exercises the CF-named
    variables, the QAQC flags, and `sigmat` — a property omnisea does not curate, which must
    still travel.
    """
    return load("onc_scalardata.json")


def a_match(location_code="FGPPN", name="Folger Pinnacle"):
    return StationMatch(
        source="onc_scalardata", provider="onc", station_id=location_code, name=name,
        lat=48.8145, lon=-125.2825, extra={"location_code": location_code},
    )


def query(**options):
    return Query.from_position(**FOLGER, time=DAY, radius_km=10, onc_token=TOKEN, **options)


# --------------------------------------------------------------------------- the credential


class TestToken:
    def test_a_missing_token_says_how_to_get_one(self, provider, monkeypatch):
        # Explicitly unset: this must hold for a developer who *does* have ONC_TOKEN exported,
        # which is exactly the person most likely to run it.
        monkeypatch.delenv("ONC_TOKEN", raising=False)
        bare = Query.from_position(**FOLGER, time=DAY, radius_km=10)
        with pytest.raises(ProviderError) as excinfo:
            provider.token(bare)
        message = str(excinfo.value)
        assert "data.oceannetworks.ca" in message
        assert "ONC_TOKEN" in message
        assert "onc_token=" in message

    def test_the_environment_is_used_when_no_option_is_given(self, provider, monkeypatch):
        monkeypatch.setenv("ONC_TOKEN", "from-the-environment")
        bare = Query.from_position(**FOLGER, time=DAY, radius_km=10)
        assert provider.token(bare) == "from-the-environment"

    def test_the_explicit_option_wins_over_the_environment(self, provider, monkeypatch):
        monkeypatch.setenv("ONC_TOKEN", "from-the-environment")
        assert provider.token(query()) == TOKEN

    def test_asking_for_onc_by_name_without_a_token_fails_loudly(self, source, monkeypatch):
        """Silence would read as "ONC has nothing near you" — a different and wrong conclusion
        from "you have not authenticated"."""
        monkeypatch.delenv("ONC_TOKEN", raising=False)
        with pytest.raises(ProviderError, match="requires an API token"):
            source.discover(Query.from_position(**FOLGER, time=DAY, radius_km=10,
                                                providers=["onc_scalardata"]))

    def test_a_plain_query_skips_onc_quietly(self, source, monkeypatch):
        """Most users have no ONC token. Raising here would print a failure on every single
        omnisea call they make, for a source they never asked for."""
        monkeypatch.delenv("ONC_TOKEN", raising=False)
        assert source.discover(Query.from_position(**FOLGER, time=DAY, radius_km=10)) == []

    def test_the_whole_provider_named_also_counts_as_asking(self, source, monkeypatch):
        monkeypatch.delenv("ONC_TOKEN", raising=False)
        with pytest.raises(ProviderError, match="requires an API token"):
            source.discover(Query.from_position(**FOLGER, time=DAY, radius_km=10,
                                                providers=["onc"]))


class TestTheTokenNeverEscapes:
    """ONC authenticates in the query string, and a URL like that outlives the request."""

    def test_it_is_redacted_from_a_url(self):
        url = f"https://data.oceannetworks.ca/api/scalardata?locationCode=FGPD&token={TOKEN}"
        assert TOKEN not in redact_url(url)
        assert "token=REDACTED" in redact_url(url)

    def test_a_url_with_no_query_is_untouched(self):
        assert redact_url("https://example.org/a/b") == "https://example.org/a/b"

    def test_it_is_scrubbed_from_an_echoed_error_body(self, monkeypatch):
        """ONC quotes a corrected URL — with your token in it — inside its 400 body."""
        from omnisea import http

        class Response:
            ok = False
            status_code = 400
            url = f"https://data.oceannetworks.ca/api/scalardata?token={TOKEN}"
            text = ""

            def json(self):
                return {"errors": [{"errorMessage": f"try https://x/api?token={TOKEN}"}]}

        monkeypatch.setattr(http, "get_session", lambda: type(
            "S", (), {"get": staticmethod(lambda *a, **k: Response())})())
        with pytest.raises(UpstreamError) as excinfo:
            http.get_json("https://data.oceannetworks.ca/api/scalardata", {"token": TOKEN})
        assert TOKEN not in str(excinfo.value), "the token came back in the error body"

    def test_the_recorded_source_url_carries_no_token(self, source, scalardata, monkeypatch):
        monkeypatch.setattr(source.provider, "api", lambda *a, **k: scalardata)
        monkeypatch.setattr(source, "_categories_for", lambda q, m: ["CTD"])
        (series,) = source.fetch(query(), [a_match()])
        assert TOKEN not in series.attrs["source_url"]
        assert "token" not in series.attrs["source_url"]

    def test_the_debug_log_line_is_redacted(self, caplog, monkeypatch):
        from omnisea import http

        monkeypatch.setattr(http, "get_session", lambda: type(
            "S", (), {"get": staticmethod(lambda *a, **k: (_ for _ in ()).throw(
                __import__("requests").RequestException("boom")))})())
        with caplog.at_level("DEBUG", logger="omnisea.http"), pytest.raises(UpstreamError):
            http.get_json("https://data.oceannetworks.ca/api/x", {"token": TOKEN})
        assert TOKEN not in caplog.text


# --------------------------------------------------------------------------- discovery


class TestDiscovery:
    def test_locations_are_filtered_client_side_by_distance(self, source, monkeypatch, locations):
        """/api/locations rejects lat, lon and radius by name, so omnisea filters the list."""
        monkeypatch.setattr(source.provider, "locations", lambda q, property_code=None: locations)
        matches = source.discover(
            Query.from_position(**BAMFIELD, time=DAY, radius_km=5, onc_token=TOKEN)
        )
        assert matches, "there are real ONC locations within 5 km of Bamfield"
        assert all(m.distance_km <= 5 for m in matches)

    def test_a_wider_radius_finds_more(self, source, monkeypatch, locations):
        monkeypatch.setattr(source.provider, "locations", lambda q, property_code=None: locations)
        near = source.discover(Query.from_position(**BAMFIELD, time=DAY, radius_km=5,
                                                   onc_token=TOKEN))
        far = source.discover(Query.from_position(**BAMFIELD, time=DAY, radius_km=40,
                                                  onc_token=TOKEN))
        assert len(far) > len(near)

    def test_locations_with_no_instrument_ever_are_skipped(self, source, monkeypatch, locations):
        """hasDeviceData=false means scalardata would answer with an error, not data."""
        monkeypatch.setattr(source.provider, "locations", lambda q, property_code=None: locations)
        matches = source.discover(
            Query.from_position(**BAMFIELD, time=DAY, radius_km=40, onc_token=TOKEN)
        )
        codes = {m.station_id for m in matches}
        skipped = {
            x["locationCode"] for x in locations
            if str(x.get("hasDeviceData")).lower() != "true"
        }
        assert not (codes & skipped)

    def test_a_cf_name_is_translated_to_an_onc_property_code(self, source):
        codes = source._requested_property_codes(
            Query.from_position(**FOLGER, time=DAY, radius_km=10, onc_token=TOKEN,
                                variables=["sea_water_temperature"])
        )
        assert "seawatertemperature" in codes

    def test_an_onc_property_code_is_accepted_verbatim(self, source):
        codes = source._requested_property_codes(
            Query.from_position(**FOLGER, time=DAY, radius_km=10, onc_token=TOKEN,
                                variables=["salinity"])
        )
        assert "salinity" in codes


class TestDeviceCategories:
    def test_a_requested_category_is_intersected_with_what_is_deployed(self, source, monkeypatch):
        """ONC answers a category that was never deployed with HTTP 400, so asking blindly
        across a radius would fail the whole query on the first seismometer."""
        monkeypatch.setattr(source.provider, "device_categories",
                            lambda q, code: [{"deviceCategoryCode": "ACCELEROMETER"}])
        assert source._categories_for(query(onc_device_categories="CTD"), a_match()) == []

    def test_every_deployed_category_is_used_when_none_is_named(self, source, monkeypatch):
        monkeypatch.setattr(source.provider, "device_categories", lambda q, code: [
            {"deviceCategoryCode": "CTD"}, {"deviceCategoryCode": "OXYSENSOR"}])
        assert source._categories_for(query(), a_match()) == ["CTD", "OXYSENSOR"]

    def test_an_unreadable_category_list_trusts_the_caller(self, source, monkeypatch):
        monkeypatch.setattr(source.provider, "device_categories", lambda q, code: [])
        assert source._categories_for(query(onc_device_categories="CTD"), a_match()) == ["CTD"]


# --------------------------------------------------------------------------- shaping


class TestColumnarTranspose:
    """ONC returns parallel arrays per sensor, not rows — the one adapter here that transposes."""

    @pytest.fixture
    def series(self, source, scalardata, monkeypatch):
        monkeypatch.setattr(source.provider, "api", lambda *a, **k: scalardata)
        monkeypatch.setattr(source, "_categories_for", lambda q, m: ["CTD"])
        q = Query.from_position(**FOLGER, time=FIXTURE_WINDOW, radius_km=10, onc_token=TOKEN)
        (built,) = source.fetch(q, [a_match()])
        return built

    def test_parallel_arrays_become_a_time_indexed_frame(self, series):
        assert isinstance(series.frame.index, pd.DatetimeIndex)
        assert series.frame.index.is_monotonic_increasing
        assert not series.frame.index.has_duplicates
        assert len(series.frame) > 1

    def test_properties_get_their_cf_names(self, series):
        assert "sea_water_temperature" in series.frame.columns
        assert "sea_water_practical_salinity" in series.frame.columns

    def test_units_come_from_the_sensor_not_a_table(self, series):
        """The same property is served in different units by different instruments."""
        attrs = series.var_attrs["sea_water_temperature"]
        assert attrs["units"], "units must be carried from unitOfMeasure"
        assert PROPERTY_FIELDS["seawatertemperature"].units is None

    def test_qaqc_flags_are_carried_with_their_vocabulary(self, series):
        qc = "sea_water_temperature_qc"
        assert qc in series.frame.columns
        assert "Data Passed All Tests" in series.var_attrs[qc]["comment"]

    def test_unmapped_properties_still_travel(self, series):
        carried = [
            name for name, attrs in series.var_attrs.items()
            if attrs.get("omnisea_mapped") == 0 and not name.endswith("_qc")
        ]
        assert carried, "ONC publishes 219 properties; omnisea names a fraction"

    def test_the_sensor_identity_is_recorded_on_each_variable(self, series):
        attrs = series.var_attrs["sea_water_temperature"]
        assert attrs["onc_property_code"] == "seawatertemperature"
        assert attrs["onc_sensor_code"]

    def test_the_node_path_separates_instruments_at_one_location(self, series):
        assert series.node_path == "in_situ/onc/FGPPN/CTD"


class TestCitationPassthrough:
    """ONC returns the exact wording it wants credited, plus a resolvable DOI."""

    @pytest.fixture
    def tree(self, source, scalardata, monkeypatch):
        monkeypatch.setattr(source.provider, "api", lambda *a, **k: scalardata)
        monkeypatch.setattr(source, "_categories_for", lambda q, m: ["CTD"])
        q = Query.from_position(**FOLGER, time=FIXTURE_WINDOW, radius_km=10, onc_token=TOKEN)
        return omnisea.tree.build_tree(q, source.fetch(q, [a_match()]))

    def test_the_doi_reaches_the_node(self, tree):
        node = next(n for n in tree.subtree if n.dataset.data_vars)
        assert node.dataset.attrs["doi"].startswith("10.")

    def test_oncs_own_citation_wording_is_kept(self, tree):
        node = next(n for n in tree.subtree if n.dataset.data_vars)
        assert "Ocean Networks Canada" in node.dataset.attrs["citation"]

    def test_provenance_lists_the_source(self, tree):
        assert "onc_scalardata" in omnisea.sources_used(tree)


# --------------------------------------------------------------------------- upstream quirks


class TestEmptyResultsAreNotFailures:
    @pytest.mark.parametrize("detail", [
        "A device with category CTD was deployed at location X but not during the provided "
        "time range (20240701T000000.000Z to 20240702T000000.000Z).",
        "There is no deployment of a device with category CTD at location FGPD.H1.",
    ])
    def test_a_400_meaning_no_data_is_translated(self, detail):
        assert _means_no_data(UpstreamError("x", status=400, detail=detail))

    def test_a_bad_token_still_raises(self):
        assert not _means_no_data(
            UpstreamError("x", status=401, detail="Invalid parameter value")
        )

    def test_a_genuine_server_error_still_raises(self):
        assert not _means_no_data(UpstreamError("x", status=500, detail="internal error"))

    def test_one_quiet_instrument_does_not_sink_the_query(self, source, scalardata, monkeypatch):
        """A radius covering a live CTD and a retired one must return the live one."""
        def api(path, params, *, query, source_name=None, **kwargs):
            if params.get("locationCode") == "RETIRED":
                raise UpstreamError("x", status=400,
                                    detail="not during the provided time range")
            return scalardata

        monkeypatch.setattr(source.provider, "api",
                            lambda path, params, **kw: api(path, params, query=None))
        monkeypatch.setattr(source, "_categories_for", lambda q, m: ["CTD"])
        built = source.fetch(query(), [a_match("RETIRED"), a_match("FGPPN")])
        assert [s.match.station_id for s in built] == ["FGPPN"]


class TestResampling:
    def test_the_default_is_a_minute_not_raw_seconds(self, source):
        """Raw ONC data can be 1 Hz: a day of one CTD is 86,400 rows per sensor."""
        assert source._resample_seconds(query()) == 60

    def test_raw_samples_are_available_explicitly(self, source):
        assert source._resample_seconds(query(onc_resample_seconds=0)) == 0

    def test_the_row_estimate_follows_the_resampling(self, source):
        coarse = source._estimate_rows(query(onc_resample_seconds=3600))
        assert coarse == 24, "one day at hourly resampling"
        assert source._estimate_rows(query(onc_resample_seconds=0)) == 86_400

    def test_a_nonsense_interval_is_rejected(self, source):
        with pytest.raises(QueryError, match="onc_resample_seconds"):
            source._resample_seconds(query(onc_resample_seconds="soon"))


class TestRegistration:
    def test_it_is_registered_as_a_source_and_a_provider(self):
        assert "onc" in omnisea.providers()
        assert "onc_scalardata" in omnisea.sources()

    def test_measurements_are_never_cached_but_the_catalogue_is(self):
        from omnisea.http import NEVER_CACHE, cache_policy

        policy = cache_policy()
        assert policy["data.oceannetworks.ca/api/scalardata*"] == NEVER_CACHE
        assert policy["data.oceannetworks.ca/api/locations*"] != NEVER_CACHE


# --------------------------------------------------------------------------- live


@live
@needs_token
class TestLive:
    def test_folger_passage_returns_a_real_ctd_series(self):
        tree = omnisea.fetch(**FOLGER, radius_km=6, time=DAY,
                             providers="onc_scalardata", onc_device_categories="CTD")
        summary = omnisea.summary(tree)
        assert not summary.empty, "Folger Passage is a cabled ONC node"
        assert (summary["n_time"] > 0).all()

        temperature = omnisea.fields(tree).query("variable == 'sea_water_temperature'")
        assert not temperature.empty
        # Barkley Sound at depth, in July: cold but not freezing.
        node = next(n for n in tree.subtree if "sea_water_temperature" in n.dataset.data_vars)
        values = node.dataset["sea_water_temperature"].to_series().dropna()
        assert 2.0 < values.min() and values.max() < 20.0, f"implausible: {values.describe()}"

    def test_the_live_citation_carries_a_resolvable_doi(self):
        tree = omnisea.fetch(**FOLGER, radius_km=6, time=DAY,
                             providers="onc_scalardata", onc_device_categories="CTD")
        node = next(n for n in tree.subtree if n.dataset.data_vars)
        assert node.dataset.attrs.get("doi", "").startswith("10.")

    def test_no_token_reaches_the_result(self):
        token = os.environ["ONC_TOKEN"]
        tree = omnisea.fetch(**FOLGER, radius_km=6, time=DAY,
                             providers="onc_scalardata", onc_device_categories="CTD")
        blob = json.dumps({n.path: {k: str(v) for k, v in n.dataset.attrs.items()}
                           for n in tree.subtree})
        assert token not in blob
        assert token not in omnisea.citation(tree)
