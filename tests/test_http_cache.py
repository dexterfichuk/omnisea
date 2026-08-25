"""The optional response cache: what it stores, what it refuses to store, and the swap itself.

Nothing here touches the network. A counting transport adapter stands in for the upstream, which
means the assertions are made against the *real* :data:`omnisea.http.CACHE_POLICY` and real URLs
— when a request does not reach the adapter, it genuinely came out of the cache.
"""

from __future__ import annotations

import io
import subprocess
import sys
from datetime import timedelta

import pytest
import requests
import urllib3
from requests.adapters import HTTPAdapter

from omnisea import http
from omnisea.errors import MissingDependencyError, OmniseaError

IWLS = "https://api-iwls.dfo-mpo.gc.ca/api/v1"
STATIONS = f"{IWLS}/stations"
STATION_METADATA = f"{IWLS}/stations/5cebf1df3d0f4a073c4bc0f6/metadata"
OBSERVATIONS = f"{IWLS}/stations/5cebf1df3d0f4a073c4bc0f6/data?time-series-code=wlo"
GEOMET = "https://api.weather.gc.ca/collections"
CLIMATE_STATIONS = f"{GEOMET}/climate-stations/items?bbox=1,2,3,4"
CLIMATE_HOURLY = f"{GEOMET}/climate-hourly/items?STATION_ID=1707"
AHCCD_STATIONS = f"{GEOMET}/ahccd-stations/items?bbox=1,2,3,4"
AHCCD_ANNUAL = f"{GEOMET}/ahccd-annual/items?ID=1"
SWOB = f"{GEOMET}/swob-realtime/items?bbox=1,2,3,4"
ELSEWHERE = "https://example.org/some/other/api"


class CountingAdapter(HTTPAdapter):
    """A transport that answers everything with ``{"n": <call number>}`` and remembers who asked.

    The body changes on every call, so a test that gets the same body twice has proved reuse
    rather than coincidence.
    """

    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    def send(self, request, **kwargs):
        self.calls.append(request.url)
        raw = urllib3.HTTPResponse(
            body=io.BytesIO(f'{{"n": {len(self.calls)}}}'.encode()),
            headers={"Content-Type": "application/json"},
            status=200,
            preload_content=False,
        )
        return self.build_response(request, raw)

    def count(self, url: str) -> int:
        return self.calls.count(url)


@pytest.fixture(autouse=True)
def isolated_session():
    """Give each test its own module session and hand the original back afterwards."""
    original = http._session
    http._session = None
    yield
    http.disable_cache()
    http._session = original


@pytest.fixture
def requests_cache():
    return pytest.importorskip("requests_cache")


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "omnisea-http"


@pytest.fixture
def enabled(requests_cache, cache_path):
    """A cache-backed session, still wired to the real transport."""
    return http.enable_cache(path=cache_path)


@pytest.fixture
def cached(requests_cache, cache_path):
    """A cache-backed session plus the stub transport underneath it.

    Mounting the stub replaces the retry adapter, so tests that care about the adapter itself
    use ``enabled`` instead.
    """

    def _enable(**kwargs):
        session = http.enable_cache(path=cache_path, **kwargs)
        adapter = CountingAdapter()
        session.mount("https://", adapter)
        return session, adapter

    return _enable


# --------------------------------------------------------------------------- off by default


def test_cache_is_off_by_default():
    session = http.get_session()
    assert isinstance(session, requests.Session)
    assert not hasattr(session, "cache"), "caching must be opt-in"


def test_omnisea_works_without_the_extra_installed():
    # Run out-of-process so the import really happens with requests-cache unavailable: a
    # top-level import of the optional package would break `import omnisea` for everyone who
    # never asked for caching.
    script = (
        "import sys; sys.modules['requests_cache'] = None\n"
        "import omnisea\n"
        "assert omnisea.http.get_session().headers['User-Agent'].startswith('omnisea/')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- enable / disable


def test_enable_cache_swaps_the_module_session(cached):
    session, _ = cached()
    assert http.get_session() is session
    assert hasattr(session, "cache")


def test_disable_cache_restores_a_plain_session(cached):
    session, _ = cached()
    http.disable_cache()
    fresh = http.get_session()
    assert fresh is not session
    assert not hasattr(fresh, "cache")


def test_disable_cache_leaves_an_uncached_session_alone():
    session = http.get_session()
    http.disable_cache()
    assert http.get_session() is session


def test_disable_cache_with_clear_empties_the_store(cached):
    session, _ = cached()
    session.get(STATIONS)
    assert list(session.cache.urls()) == [STATIONS]

    http.disable_cache(clear=True)
    session, adapter = cached()
    session.get(STATIONS)
    assert adapter.count(STATIONS) == 1, "a cleared cache must go back to the network"


# --------------------------------------------------------------------------- missing extra


def test_missing_dependency_names_the_extra(monkeypatch, cache_path):
    # A None entry in sys.modules is what an uninstalled package looks like to an import.
    monkeypatch.setitem(sys.modules, "requests_cache", None)

    with pytest.raises(OmniseaError) as excinfo:
        http.enable_cache(path=cache_path)

    message = str(excinfo.value)
    assert "requests-cache" in message
    assert 'pip install "omnisea[cache]"' in message
    # Deliberately also an ImportError, so code probing for optional dependencies with
    # `except ImportError` keeps working. What must not escape is an *unwrapped*
    # ModuleNotFoundError naming a package the user never asked for by name.
    assert isinstance(excinfo.value, MissingDependencyError)
    assert not isinstance(excinfo.value, ModuleNotFoundError)
    assert http._session is None, "a failed enable must not leave a half-swapped session"


# --------------------------------------------------------------------------- the swap keeps


def test_cached_session_keeps_the_retry_adapter(enabled):
    adapter = enabled.get_adapter("https://api-iwls.dfo-mpo.gc.ca/")
    retry = adapter.max_retries
    assert retry.total == 4
    assert 429 in retry.status_forcelist
    assert retry.backoff_factor == 0.7
    assert adapter.poolmanager.connection_pool_kw.get("maxsize") == 32


def test_cached_session_keeps_the_user_agent(enabled):
    assert enabled.headers["User-Agent"].startswith("omnisea/")
    assert enabled.headers["Accept"] == "application/json"


def test_get_json_goes_through_the_cached_session(cached):
    # get_json() holds the concurrency semaphore around the call, so this also proves the cache
    # sits inside that discipline rather than beside it.
    _, adapter = cached()
    assert http.get_json(STATIONS) == {"n": 1}
    assert http.get_json(STATIONS) == {"n": 1}
    assert adapter.count(STATIONS) == 1


# --------------------------------------------------------------------------- the policy


def test_station_list_is_fetched_once_and_reused(cached):
    session, adapter = cached()
    first = session.get(STATIONS)
    second = session.get(STATIONS)

    assert adapter.count(STATIONS) == 1
    assert first.json() == second.json() == {"n": 1}
    assert second.from_cache is True


def test_station_metadata_is_reused(cached):
    session, adapter = cached()
    session.get(STATION_METADATA)
    session.get(STATION_METADATA)
    assert adapter.count(STATION_METADATA) == 1


def test_station_catalogues_of_other_providers_are_reused(cached):
    session, adapter = cached()
    session.get(CLIMATE_STATIONS)
    session.get(CLIMATE_STATIONS)
    assert adapter.count(CLIMATE_STATIONS) == 1


def test_observations_are_never_served_from_cache(cached):
    session, adapter = cached()
    first = session.get(OBSERVATIONS)
    second = session.get(OBSERVATIONS)

    assert adapter.count(OBSERVATIONS) == 2, "a stale water level is a wrong number"
    assert first.json() != second.json()
    assert list(session.cache.urls()) == []


def test_the_climate_archive_is_reused(cached):
    session, adapter = cached()
    session.get(CLIMATE_HOURLY)
    session.get(CLIMATE_HOURLY)
    assert adapter.count(CLIMATE_HOURLY) == 1


def test_ahccd_stations_get_the_catalogue_expiry_not_the_annual_one(cached):
    # '*/collections/*-stations/items' has to be matched before '*/collections/ahccd-*/items',
    # and rules are tried in order, so this pins the ordering rather than trusting it.
    session, _ = cached()
    for url in (AHCCD_STATIONS, AHCCD_ANNUAL):
        session.get(url)

    stations = session.get(AHCCD_STATIONS)
    annual = session.get(AHCCD_ANNUAL)
    assert stations.from_cache and annual.from_cache
    assert stations.expires - annual.expires > timedelta(days=5)


def test_unmatched_urls_are_not_cached_by_default(cached):
    session, adapter = cached()
    session.get(ELSEWHERE)
    session.get(ELSEWHERE)
    assert adapter.count(ELSEWHERE) == 2


def test_realtime_stays_uncached_even_when_caching_everything_else(cached):
    # The volatile patterns are in the policy explicitly so that they survive a caller who opts
    # the rest of the world in.
    session, adapter = cached(expire_after=3600)

    session.get(ELSEWHERE)
    session.get(ELSEWHERE)
    assert adapter.count(ELSEWHERE) == 1

    session.get(SWOB)
    session.get(SWOB)
    assert adapter.count(SWOB) == 2


def test_a_third_party_providers_policy_is_honoured(cached):
    # The endpoint judgment lives on the provider, so a plugin declares cacheability the same
    # way the built-ins do — no core edit, no enable_cache() argument.
    from datetime import timedelta

    from omnisea import registry
    from omnisea.providers.base import Provider, RetrievalSource

    class PluginSource(RetrievalSource):
        name = "plugin_sst"
        title = "Plugin SST"
        node_path = "in_situ/plugin"

        def discover(self, query):
            return []

        def fetch(self, query, matches):
            return []

    class PluginProvider(Provider):
        name = "plugin"
        title = "Plugin Network"
        base_url = "https://plugin.example.org"
        license = "CC-BY-4.0"
        cache_policy = {
            "plugin.example.org/api/*/data": http.NEVER_CACHE,
            "plugin.example.org/api/stations": timedelta(days=7),
        }

        def build_sources(self):
            return [PluginSource(self)]

    registry.register_provider(PluginProvider(), replace=True)
    try:
        session, adapter = cached()
        stations = "https://plugin.example.org/api/stations"
        data = "https://plugin.example.org/api/BAM01/data"
        for url in (stations, stations, data, data):
            session.get(url)
        assert adapter.count(stations) == 1, "the plugin's catalogue rule must apply"
        assert adapter.count(data) == 2, "the plugin's never-cache rule must apply"
    finally:
        registry._SOURCES.pop("plugin_sst", None)
        registry._PROVIDERS.pop("plugin", None)


def test_caller_rules_take_precedence_over_the_policy(cached):
    session, adapter = cached(
        urls_expire_after={"api-iwls.dfo-mpo.gc.ca/api/v1/stations": http.NEVER_CACHE}
    )
    session.get(STATIONS)
    session.get(STATIONS)
    assert adapter.count(STATIONS) == 2


# --------------------------------------------------------------------------- persistence


@pytest.mark.network
def test_live_station_list_is_cached_despite_the_upstream_no_store(enabled):
    # IWLS answers everything with "no-cache, no-store, max-age=0, must-revalidate". The stub
    # transport above cannot catch a regression that started honouring that header, and doing so
    # would silently reduce the whole feature to a no-op — so this one runs against the service.
    # `?code=` keeps it to a single station instead of the 2 MB catalogue.
    first = enabled.get(STATIONS, params={"code": "08545"}, timeout=(10, 60))
    second = enabled.get(STATIONS, params={"code": "08545"}, timeout=(10, 60))

    assert "no-store" in first.headers.get("Cache-Control", "")
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.json() == first.json()


def test_cache_survives_a_new_session_on_the_same_path(cached):
    session, adapter = cached()
    assert session.get(STATIONS).json() == {"n": 1}

    http.disable_cache()
    session, adapter = cached()
    response = session.get(STATIONS)

    assert adapter.calls == [], "the second process must not re-fetch the station list"
    assert response.json() == {"n": 1}
    assert response.from_cache is True


def test_memory_backend_keeps_nothing_after_disable(requests_cache):
    def stub():
        session = http.enable_cache(backend="memory")
        adapter = CountingAdapter()
        session.mount("https://", adapter)
        return session, adapter

    session, _ = stub()
    session.get(STATIONS)
    http.disable_cache()

    session, adapter = stub()
    session.get(STATIONS)
    assert adapter.count(STATIONS) == 1, "the memory backend must not outlive the session"
