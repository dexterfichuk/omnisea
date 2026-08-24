"""Live integration tests. Run with `pytest -m network`.

These exist to catch the things fixtures cannot: upstream contract changes, the interval caps,
paging behaviour, and whether the numbers still describe real tides.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import omnisea
from omnisea.errors import UpstreamError
from omnisea.http import get_json, paginate_ogc_items
from omnisea.providers.dfo import BASE as IWLS_BASE

pytestmark = pytest.mark.network

# Bamfield Marine Sciences Centre, Barkley Sound — the running example.
BAMFIELD = dict(lat=48.8353, lon=-125.1358, radius_km=30)
BAMFIELD_TIDE_STATION = "08545"
WEEK = ("2024-07-01", "2024-07-08")
VICTORIA_IWLS_ID = "5cebf1df3d0f4a073c4bbd1e"


# --------------------------------------------------------------------------- IWLS limits


class TestIwlsIntervalCaps:
    def test_one_minute_over_seven_days_is_a_clean_omnisea_error(self):
        """The upstream 400 must surface as an omnisea error carrying the server's own words."""
        with pytest.raises(UpstreamError) as excinfo:
            get_json(
                f"{IWLS_BASE}/stations/{VICTORIA_IWLS_ID}/data",
                {
                    "time-series-code": "wlo",
                    "from": "2024-07-01T00:00:00Z",
                    "to": "2024-07-20T00:00:00Z",
                    "resolution": "ONE_MINUTE",
                },
                provider="dfo_tides",
            )
        assert excinfo.value.status == 400
        assert "7 days" in (excinfo.value.detail or "")

    def test_coarse_resolution_allows_thirty_one_days(self):
        """The cap is resolution-dependent, which is why chunking must be too."""
        rows = get_json(
            f"{IWLS_BASE}/stations/{VICTORIA_IWLS_ID}/data",
            {
                "time-series-code": "wlo",
                "from": "2024-07-01T00:00:00Z",
                "to": "2024-07-31T00:00:00Z",
                "resolution": "SIXTY_MINUTES",
            },
            provider="dfo_tides",
        )
        assert len(rows) > 700

    def test_thirty_two_days_exceeds_even_the_coarse_cap(self):
        with pytest.raises(UpstreamError) as excinfo:
            get_json(
                f"{IWLS_BASE}/stations/{VICTORIA_IWLS_ID}/data",
                {
                    "time-series-code": "wlo",
                    "from": "2024-07-01T00:00:00Z",
                    "to": "2024-08-05T00:00:00Z",
                    "resolution": "FIFTEEN_MINUTES",
                },
                provider="dfo_tides",
            )
        assert "31 days" in (excinfo.value.detail or "")


class TestIwlsChunkingAcrossTheBoundary:
    def test_a_window_longer_than_the_cap_is_stitched_without_gaps_or_duplicates(self):
        """20 days at ONE_MINUTE needs three chunks; the seams must be invisible."""
        tree = omnisea.fetch(
            lat=48.8353, lon=-125.1358, radius_km=2,
            time=("2024-07-01", "2024-07-21"),
            providers="dfo_tides", resolution="ONE_MINUTE", series=["wlo"],
            max_rows=100_000,
        )
        ds = tree[f"/in_situ/tides/{BAMFIELD_TIDE_STATION}"].dataset
        time = pd.DatetimeIndex(ds["time"].values)
        assert time.is_unique, "chunk boundaries were not de-duplicated"
        assert time.is_monotonic_increasing
        gaps = time.to_series().diff().dropna()
        assert gaps.max() <= pd.Timedelta(minutes=60), "a chunk boundary left a hole"


# --------------------------------------------------------------------------- pygeoapi


class TestOgcPaging:
    def test_paging_walks_past_a_single_page(self):
        """pygeoapi caps `limit`, so more than one page must actually be requested."""
        url = "https://api.weather.gc.ca/collections/climate-hourly/items"
        params = {
            "CLIMATE_IDENTIFIER": "1018611",
            "datetime": "2024-01-01T00:00:00Z/2024-03-01T00:00:00Z",
        }
        got = list(paginate_ogc_items(url, params, provider="eccc_climate", page_size=100))
        assert len(got) > 100
        ids = [f["properties"]["ID"] for f in got]
        assert len(ids) == len(set(ids)), "paging returned duplicate rows"

    def test_ceiling_raises_rather_than_truncating_silently(self):
        from omnisea.errors import PayloadTooLargeError

        url = "https://api.weather.gc.ca/collections/climate-hourly/items"
        with pytest.raises(PayloadTooLargeError):
            list(paginate_ogc_items(url, {}, provider="eccc_climate", max_items=50))


# --------------------------------------------------------------------------- discovery


class TestDiscovery:
    def test_bamfield_gauge_is_found_at_the_research_station(self):
        catalog = omnisea.discover(**BAMFIELD, time=WEEK, providers="dfo_tides")
        nearest = min(catalog, key=lambda m: m.distance_km)
        assert nearest.station_id == BAMFIELD_TIDE_STATION
        assert nearest.distance_km < 1.0

    def test_stations_without_hourly_records_are_excluded(self):
        """climate-stations lists every station; HLY_FIRST_DATE null means no hourly data."""
        catalog = omnisea.discover(**BAMFIELD, time=WEEK, providers="eccc_climate")
        assert all(m.first is not None for m in catalog)

    def test_a_source_with_nothing_to_offer_returns_empty_not_an_error(self):
        catalog = omnisea.discover(**BAMFIELD, time=WEEK)
        assert catalog.errors == {}

    def test_station_with_no_overlap_is_absent_from_the_tree(self):
        """hydrometric-realtime holds ~30 days, so a 2024 window must exclude those gauges."""
        tree = omnisea.fetch(**BAMFIELD, time=WEEK, providers="eccc_hydrometric")
        assert not [n for n in tree.subtree if n.dataset.data_vars]


# --------------------------------------------------------------------------- end to end


class TestBamfieldEndToEnd:
    @pytest.fixture(scope="class")
    def tree(self):
        return omnisea.fetch(**BAMFIELD, time=WEEK, nearest=1)

    def test_tide_and_weather_arrive_in_separate_branches(self, tree):
        paths = {n.path for n in tree.subtree if n.dataset.data_vars}
        assert f"/in_situ/tides/{BAMFIELD_TIDE_STATION}" in paths
        assert f"/predictions/tides_hilo/{BAMFIELD_TIDE_STATION}" in paths

    def test_observations_are_never_merged_with_predictions(self, tree):
        obs = tree[f"/in_situ/tides/{BAMFIELD_TIDE_STATION}"].dataset
        pred = tree[f"/predictions/tides_hilo/{BAMFIELD_TIDE_STATION}"].dataset
        assert set(obs.data_vars) != set(pred.data_vars)

    def test_datum_offsets_are_recorded_on_the_tide_node(self, tree):
        attrs = tree[f"/in_situ/tides/{BAMFIELD_TIDE_STATION}"].attrs
        assert "datum_offset_CGVD2013" in attrs
        assert attrs["datum"] == "chart datum (CD)"

    def test_the_window_is_honoured_exactly(self, tree):
        for node in tree.subtree:
            ds = node.dataset
            if not ds.data_vars or "time" not in ds.coords:
                continue
            times = pd.DatetimeIndex(ds["time"].values)
            assert times.min() >= pd.Timestamp("2024-07-01")
            assert times.max() <= pd.Timestamp("2024-07-08")

    def test_netcdf_round_trip_preserves_the_group_structure(self, tree, tmp_path):
        pytest.importorskip("netCDF4")
        path = tmp_path / "bamfield.nc"
        tree.to_netcdf(path)
        back = xr.open_datatree(path)
        assert {n.path for n in tree.subtree if n.dataset.data_vars} == {
            n.path for n in back.subtree if n.dataset.data_vars
        }

    def test_summary_and_dataframe_agree_on_the_stations(self, tree):
        frame = omnisea.to_dataframe(tree)
        assert set(frame["station_id"]) == set(omnisea.summary(tree)["station_id"])


class TestScientificSanity:
    """Parsing correctly is not the same as being right. These check the physics."""

    @pytest.fixture(scope="class")
    def tides(self):
        tree = omnisea.fetch(
            lat=48.8353, lon=-125.1358, radius_km=2, time=WEEK,
            providers="dfo_tides", series=["wlo", "wlp-hilo"],
        )
        node = f"/in_situ/tides/{BAMFIELD_TIDE_STATION}"
        hilo = f"/predictions/tides_hilo/{BAMFIELD_TIDE_STATION}"
        return (
            tree[node].dataset["water_surface_height_above_reference_datum"].to_series(),
            tree[hilo].dataset[
                "water_surface_height_above_reference_datum_at_extremum"
            ].to_series(),
        )

    def test_the_observed_series_is_semidiurnal(self, tides):
        """Two highs a day is the defining signature of this coast; anything else is a bug."""
        wlo, _ = tides
        v = wlo.values
        highs = ((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])).sum()
        days = (wlo.index[-1] - wlo.index[0]) / pd.Timedelta(days=1)
        assert 1.5 <= highs / days <= 3.0

    def test_water_levels_are_physically_plausible(self, tides):
        wlo, _ = tides
        assert -1.0 < wlo.min() < wlo.max() < 6.0  # metres above chart datum

    def test_predicted_extrema_line_up_with_observed_peaks(self, tides):
        """The strongest end-to-end check: two independent series must agree."""
        wlo, hilo = tides
        v = wlo.values
        peak_times = wlo.index[1:-1][(v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])]
        highs = hilo[hilo > hilo.median()]
        offsets = [
            min(abs((p - h).total_seconds()) / 60 for p in peak_times) for h in highs.index
        ]
        assert np.median(offsets) < 45, "predicted highs do not coincide with observed peaks"

    def test_predicted_and_observed_heights_agree(self, tides):
        wlo, hilo = tides
        interpolated = (
            wlo.reindex(wlo.index.union(hilo.index)).interpolate().reindex(hilo.index)
        )
        assert (interpolated - hilo).abs().mean() < 0.5  # metres
        assert interpolated.corr(hilo) > 0.95


# --------------------------------------------------------------------------- CF vocabulary


class TestCfVocabulary:
    def test_every_emitted_standard_name_is_in_the_cf_table(self):
        """`sea_surface_height_above_reference_datum` looks official and is not in the table."""
        import requests

        url = (
            "https://cfconventions.org/Data/cf-standard-names/current/src/"
            "cf-standard-name-table.xml"
        )
        table = requests.get(url, timeout=120).text
        valid = set(re.findall(r'<entry id="([^"]+)"', table))
        assert len(valid) > 4000, "did not get a plausible CF standard name table"

        offenders = []
        for source in omnisea.registry.all_sources():
            for raw, spec in source.fields.items():
                if spec.standard_name and spec.standard_name not in valid:
                    offenders.append(f"{source.name}.{raw} -> {spec.standard_name}")
        assert not offenders, "not CF standard names: " + ", ".join(offenders)
