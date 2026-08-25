"""Rolling-archive retention, and what happens when a source fails mid-fetch.

Both exist to stop the same thing: a result that looks like an answer but is really a silence.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest
import xarray as xr

import omnisea
from omnisea.catalog import Catalog
from omnisea.providers.base import (
    Provider,
    RetrievalSource,
    StationMatch,
    StationSeries,
)
from omnisea.query import Query, Site

BAMFIELD = Site(48.8353, -125.1358, "Bamfield", radius_km=30)


class _FakeProvider(Provider):
    name = "fake"
    title = "Fake Provider"
    base_url = "https://example.invalid"
    license = "none"

    def build_sources(self):
        return []


class _RollingSource(RetrievalSource):
    """A realtime collection that keeps only 30 days."""

    name = "fake_realtime"
    title = "Fake realtime"
    node_path = "in_situ/fake"
    retention = pd.Timedelta(days=30)

    def __init__(self, provider, frame=None, boom=False):
        super().__init__(provider)
        self.frame = frame
        self.boom = boom
        self.discover_calls = 0

    def discover(self, query):
        self.discover_calls += 1
        return [self.new_match(station_id="F1", name="Fake", lat=48.8353, lon=-125.1358)]

    def fetch(self, query, matches):
        if self.boom:
            raise omnisea.UpstreamError("upstream exploded", provider=self.name, status=503)
        index = pd.date_range(query.start, periods=3, freq="h", name="time")
        default = pd.DataFrame({"v": [1.0, 2, 3]}, index=index)
        frame = self.frame if self.frame is not None else default
        return [
            StationSeries(match=m, frame=frame, node_path=f"{self.node_path}/{m.station_id}")
            for m in matches
        ]


class _ArchiveSource(_RollingSource):
    """The same data, but a full historical archive."""

    name = "fake_archive"
    retention = None


def _window(start, end):
    return Query.from_sites([BAMFIELD], (start, end))


class TestRetention:
    def test_a_full_archive_reports_no_gap(self):
        source = _ArchiveSource(_FakeProvider())
        assert source.retention_gap(_window("2024-07-01", "2024-07-08")) is None
        assert source.covers(_window("2024-07-01", "2024-07-08"))

    def test_a_window_entirely_before_the_rolling_window_is_explained(self):
        """'No results' would read as 'there is no station here' — a different claim."""
        source = _RollingSource(_FakeProvider())
        gap = source.retention_gap(_window("2024-07-01", "2024-07-08"))
        assert gap is not None
        assert "30 days" in gap
        assert "entirely outside" in gap

    def test_such_a_source_is_not_even_called(self):
        source = _RollingSource(_FakeProvider())
        assert not source.covers(_window("2024-07-01", "2024-07-08"))

    def test_a_recent_window_is_covered(self):
        source = _RollingSource(_FakeProvider())
        now = pd.Timestamp.now(tz="UTC")
        recent = _window(now - pd.Timedelta(days=3), now)
        assert source.covers(recent)
        assert source.retention_gap(recent) is None

    def test_a_window_straddling_the_cutoff_is_queried_but_flagged(self):
        source = _RollingSource(_FakeProvider())
        now = pd.Timestamp.now(tz="UTC")
        straddling = _window(now - pd.Timedelta(days=90), now)
        assert source.covers(straddling), "the recent part is still available"
        assert "earlier part" in source.retention_gap(straddling)

    def test_the_note_names_the_date_it_reaches_back_to(self):
        source = _RollingSource(_FakeProvider())
        cutoff = source.retention_cutoff()
        assert cutoff is not None
        assert cutoff.strftime("%Y-%m-%d") in source.retention_gap(
            _window("2020-01-01", "2020-02-01")
        )


class TestCatalogNotes:
    def test_notes_survive_filtering(self):
        notes = {"fake_realtime": "holds 30 days"}
        catalog = Catalog(_window("2024-07-01", "2024-07-08"), [], {}, notes)
        assert catalog.filter(source="anything").notes == notes

    def test_an_empty_catalogue_explains_itself_rather_than_shrugging(self):
        catalog = Catalog(
            _window("2024-07-01", "2024-07-08"), [], {}, {"fake_realtime": "holds only 30 days"}
        )
        text = repr(catalog)
        assert "no stations found" in text
        assert "holds only 30 days" in text

    def test_an_empty_catalogue_with_nothing_to_explain_suggests_next_steps(self):
        text = repr(Catalog(_window("2024-07-01", "2024-07-08"), []))
        assert "radius_km" in text

    def test_notes_appear_alongside_results(self):
        match = StationMatch(source="s", provider="p", station_id="A", name="A",
                             lat=48.8, lon=-125.1)
        catalog = Catalog(_window("2024-07-01", "2024-07-08"), [match], {},
                          {"fake_realtime": "holds only 30 days"})
        assert "holds only 30 days" in repr(catalog)


class TestFetchErrorPolicy:
    def _catalog(self, boom_source):
        query = _window("2024-07-01", "2024-07-08")
        match = boom_source.new_match(station_id="F1", name="Fake", lat=48.8353, lon=-125.1358)
        omnisea.register_source(boom_source, replace=True)
        return Catalog(query, [match])

    def teardown_method(self):
        from omnisea import registry

        registry._SOURCES.pop("fake_realtime", None)
        registry._PROVIDERS.pop("fake", None)

    def test_a_failing_source_raises_by_default(self):
        """A tree quietly missing a source looks like a tree where it had nothing to say."""
        catalog = self._catalog(_RollingSource(_FakeProvider(), boom=True))
        with pytest.raises(omnisea.UpstreamError):
            catalog.fetch()

    def test_collect_keeps_going_and_records_the_failure(self, caplog):
        catalog = self._catalog(_RollingSource(_FakeProvider(), boom=True))
        with caplog.at_level(logging.WARNING, logger="omnisea.catalog"):
            tree = catalog.fetch(on_error="collect")
        assert isinstance(tree, xr.DataTree)
        assert "fake_realtime" in tree.attrs["omnisea_fetch_errors"]
        assert tree.attrs["omnisea_fetch_incomplete"] == 1
        assert "fetch failed" in caplog.text

    def test_a_complete_fetch_carries_no_incomplete_marker(self):
        catalog = self._catalog(_RollingSource(_FakeProvider(), boom=False))
        tree = catalog.fetch()
        assert "omnisea_fetch_incomplete" not in tree.attrs

    def test_an_unknown_policy_is_rejected(self):
        catalog = self._catalog(_RollingSource(_FakeProvider(), boom=False))
        with pytest.raises(ValueError, match="on_error"):
            catalog.fetch(on_error="ignore")

    def test_the_incomplete_marker_survives_a_netcdf_round_trip(self, tmp_path):
        pytest.importorskip("netCDF4")
        catalog = self._catalog(_RollingSource(_FakeProvider(), boom=True))
        tree = catalog.fetch(on_error="collect")
        path = tmp_path / "partial.nc"
        tree.to_netcdf(path)
        assert xr.open_datatree(path).attrs["omnisea_fetch_incomplete"] == 1


class TestPeriodTrimming:
    """A period aggregate belongs to a window when its interval overlaps it.

    Not when its label instant happens to fall inside. Daily summaries are stamped at 00:00Z,
    so a request starting at noon would otherwise silently return one fewer day than the same
    request starting at midnight.
    """

    def _source(self, period):
        from omnisea.providers.eccc import EcccProvider
        from omnisea.providers.ogc import OgcFeaturesSource

        class _Periodic(OgcFeaturesSource):
            name = "periodic"
            node_path = "in_situ/periodic"

        _Periodic.period = period
        return _Periodic(EcccProvider())

    def test_a_daily_source_grows_the_window_to_whole_days(self):
        source = self._source("D")
        start, end = source.trim_window(_window("2024-07-15T12:00", "2024-07-17T06:00"))
        assert start == pd.Timestamp("2024-07-15", tz="UTC")
        assert end.floor("D") == pd.Timestamp("2024-07-17", tz="UTC")

    def test_a_monthly_source_grows_to_whole_calendar_months(self):
        source = self._source("M")
        start, end = source.trim_window(_window("2024-07-15", "2024-09-10"))
        assert start == pd.Timestamp("2024-07-01", tz="UTC")
        assert end.floor("D") == pd.Timestamp("2024-09-30", tz="UTC")

    def test_growing_is_idempotent_so_a_source_may_widen_then_trim(self):
        """hydrometric widens its own request and still lets the shared trim run."""
        source = self._source("M")
        query = _window("2024-07-15", "2024-09-10")
        once = source.trim_window(query)
        twice = source.trim_window(query.replace(start=once[0], end=once[1]))
        assert once == twice

    def test_an_instantaneous_source_is_not_widened(self):
        source = self._source(None)
        query = _window("2024-07-15T12:00", "2024-07-17")
        assert source.trim_window(query) == (query.start, query.end)

    def test_the_period_aggregate_sources_declare_their_period(self):
        from omnisea.providers.eccc import (
            EcccClimateDaily,
            EcccClimateMonthly,
            EcccHydrometricDailyMean,
            EcccProvider,
        )

        provider = EcccProvider()
        assert EcccClimateDaily(provider).period == "D"
        assert EcccClimateMonthly(provider).period == "M"
        assert EcccHydrometricDailyMean(provider).period == "D"
        # An instantaneous series must not declare one.
        from omnisea.providers.eccc import EcccClimateHourly

        assert EcccClimateHourly(provider).period is None
