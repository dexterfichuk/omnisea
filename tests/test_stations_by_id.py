"""fetch(stations=...) — the seven-of-eight persona: "I know my station id."

Every surveyed project but one starts from an id it already knows (a hardcoded 9444090 in a
boat display, a river number in a ROMS forcing script) and none of them wants to supply a
position. What these guard: ids resolve through each source's own catalogue, ambiguity is an
error rather than a guess, and the answer contains the named stations and nothing else.
"""

from __future__ import annotations

import pytest

import omnisea
from omnisea.errors import QueryError
from omnisea.providers import ndbc as ndbc_mod
from omnisea.providers import noaa as noaa_mod


@pytest.fixture
def catalogues(monkeypatch):
    monkeypatch.setattr(
        noaa_mod.CoopsProvider, "all_stations",
        lambda self: [{"id": "9444090", "name": "Port Angeles", "lat": 48.125,
                       "lng": -123.44, "greatlakes": False}],
    )
    monkeypatch.setattr(
        ndbc_mod.NdbcProvider, "all_stations",
        lambda self: [{"id": "46088", "owner": "N", "kind": "Buoy",
                       "name": "New Dungeness", "lat": 48.334, "lon": -123.165}],
    )


class TestResolution:
    def test_an_id_resolves_to_its_own_position(self, catalogues):
        from omnisea import _resolve_stations

        sites, sources, wanted = _resolve_stations({"noaa_coops": "9444090"})
        assert sources == ["noaa_coops"]
        assert wanted == {"noaa_coops": {"9444090"}}
        (site,) = sites
        assert (site.lat, site.lon) == (48.125, -123.44)
        assert site.label == "Port Angeles"

    def test_source_qualified_strings_work_too(self, catalogues):
        from omnisea import _resolve_stations

        sites, sources, wanted = _resolve_stations(["ndbc_stdmet:46088"])
        assert sources == ["ndbc_stdmet"]
        assert wanted == {"ndbc_stdmet": {"46088"}}

    def test_a_bare_id_is_refused_as_ambiguous(self):
        from omnisea import _resolve_stations

        with pytest.raises(QueryError, match="ambiguous"):
            _resolve_stations(["9444090"])

    def test_an_unknown_station_names_the_alternative(self, catalogues):
        from omnisea import _resolve_stations

        with pytest.raises(QueryError, match="spatial query"):
            _resolve_stations({"noaa_coops": "0000000"})

    def test_an_unknown_source_gets_the_registry_error(self):
        with pytest.raises(Exception, match="noaa_coops2|unknown|Unknown"):
            omnisea.discover(stations={"noaa_coops2": "9444090"},
                             time=("2024-07-01", "2024-07-02"))

    def test_stations_and_a_spatial_query_together_are_refused(self):
        with pytest.raises(QueryError, match="on its own"):
            omnisea.discover(stations={"noaa_coops": "9444090"}, lat=48.0, lon=-123.0,
                             time=("2024-07-01", "2024-07-02"))


class TestLocateHooks:
    def test_sources_without_a_catalogue_say_so(self):
        from omnisea import registry

        source = registry.get_source("eccc_climate")
        assert source.locate("1016640") is None, (
            "locate() defaults to None; the caller turns that into a QueryError "
            "naming the spatial alternative"
        )
