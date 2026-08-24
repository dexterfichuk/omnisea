"""Query construction, UTC normalization, and multi-site geometry."""

from __future__ import annotations

import pandas as pd
import pytest

from omnisea.errors import QueryError
from omnisea.query import Query, Site, as_sites, to_utc

BAMFIELD = (48.8353, -125.1358)


class TestTimeNormalization:
    def test_naive_input_is_treated_as_utc_not_local(self):
        """A naive timestamp must never pick up the developer's machine timezone."""
        ts = to_utc("2024-07-01T00:00:00")
        assert ts.tz is not None
        assert ts.tz_convert("UTC").hour == 0

    def test_aware_input_is_converted_not_relabelled(self):
        ts = to_utc("2024-07-01T00:00:00-07:00")
        assert ts == pd.Timestamp("2024-07-01T07:00:00Z")

    def test_bare_date_means_that_whole_day(self):
        q = Query.from_area((-126, 48, -125, 49), "2024-07-01")
        assert q.days == 1.0

    def test_end_before_start_is_rejected(self):
        with pytest.raises(QueryError, match="must be after"):
            Query.from_area((-126, 48, -125, 49), ("2024-07-08", "2024-07-01"))

    def test_interval_iso_is_ogc_shaped(self):
        q = Query.from_area((-126, 48, -125, 49), ("2024-07-01", "2024-07-08"))
        assert q.interval_iso == "2024-07-01T00:00:00Z/2024-07-08T00:00:00Z"


class TestBboxValidation:
    def test_inverted_latitudes_rejected(self):
        with pytest.raises(QueryError, match="north of north"):
            Query.from_area((-126, 49, -125, 48), "2024-07-01")

    def test_out_of_range_longitude_rejected(self):
        with pytest.raises(QueryError, match="longitudes"):
            Query.from_area((-200, 48, -125, 49), "2024-07-01")

    def test_antimeridian_crossing_is_refused_not_silently_wrong(self):
        with pytest.raises(QueryError, match="antimeridian"):
            Query.from_area((179, 48, -179, 49), "2024-07-01")


class TestPosition:
    def test_radius_decides_membership_not_the_bounding_box(self):
        """A point in the bbox corner but outside the circle must be excluded."""
        q = Query.from_position(*BAMFIELD, "2024-07-01", radius_km=10)
        west, south, _, _ = q.bbox
        assert not q.contains(south, west)  # corner of the box, ~14 km away
        assert q.contains(*BAMFIELD)

    def test_distance_is_measured_from_the_site(self):
        q = Query.from_position(*BAMFIELD, "2024-07-01", radius_km=30)
        assert q.distance_km(*BAMFIELD) == pytest.approx(0.0, abs=1e-6)
        assert q.distance_km(48.8353, -125.0) == pytest.approx(10.0, abs=0.6)


class TestSites:
    def test_tuple_pair_is_one_site_not_two(self):
        assert len(as_sites((48.8, -125.1))) == 1

    def test_list_of_pairs_becomes_many_sites(self):
        assert len(as_sites([(48.8, -125.1), (48.4, -123.4)])) == 2

    def test_dataframe_of_sites_is_accepted(self):
        frame = pd.DataFrame(
            {"name": ["Bamfield", "Victoria"], "lat": [48.8353, 48.42], "lon": [-125.1358, -123.37]}
        )
        sites = as_sites(frame)
        assert [s.name for s in sites] == ["Bamfield", "Victoria"]

    def test_mapping_accepts_alternate_column_names(self):
        (site,) = as_sites({"latitude": 48.8, "longitude": -125.1, "station": "BAM"})
        assert site.name == "BAM"

    def test_union_bbox_covers_every_site(self):
        q = Query.from_sites([(48.8353, -125.1358), (48.42, -123.37)], "2024-07-01", radius_km=5)
        west, south, east, north = q.bbox
        assert west < -125.13 and east > -123.37
        assert south < 48.42 and north > 48.83

    def test_point_between_distant_sites_is_excluded(self):
        """The union bbox spans the gap; the per-site radius must still reject the middle."""
        q = Query.from_sites([(48.8353, -125.1358), (48.42, -123.37)], "2024-07-01", radius_km=5)
        midpoint = (48.63, -124.25)
        assert not q.contains(*midpoint)
        assert q.contains(48.8353, -125.1358)

    def test_nearest_site_picks_the_closer_one(self):
        q = Query.from_sites(
            [Site(48.8353, -125.1358, "Bamfield"), Site(48.42, -123.37, "Victoria")],
            "2024-07-01",
            radius_km=30,
        )
        site, distance = q.nearest_site(48.84, -125.14)
        assert site.name == "Bamfield"
        assert distance < 1.0

    def test_per_site_radius_overrides_the_default(self):
        q = Query.from_sites(
            [{"lat": 48.8353, "lon": -125.1358, "name": "wide", "radius_km": 50}],
            "2024-07-01",
            radius_km=1,
        )
        assert q.sites[0].radius_km == 50

    def test_single_site_exposes_lat_lon_shorthand(self):
        q = Query.from_position(*BAMFIELD, "2024-07-01")
        assert (q.lat, q.lon) == BAMFIELD

    def test_multi_site_has_no_single_lat(self):
        q = Query.from_sites([(48.8, -125.1), (48.4, -123.4)], "2024-07-01")
        assert q.lat is None and q.is_multi_site


class TestOptions:
    def test_unknown_option_raises_rather_than_being_ignored(self):
        """A silently-dropped `resolutio=` would return minute data as if it were hourly."""
        with pytest.raises(QueryError, match="unknown option"):
            Query.from_area((-126, 48, -125, 49), "2024-07-01", resolutio="ONE_MINUTE")

    def test_typo_suggests_the_real_option(self):
        with pytest.raises(QueryError, match="did you mean 'resolution'"):
            Query.from_area((-126, 48, -125, 49), "2024-07-01", resolutin="ONE_MINUTE")

    def test_known_option_is_kept(self):
        q = Query.from_area((-126, 48, -125, 49), "2024-07-01", resolution="SIXTY_MINUTES")
        assert q.option("resolution") == "SIXTY_MINUTES"


class TestOverlaps:
    def test_open_ended_record_still_reporting_overlaps(self):
        q = Query.from_area((-126, 48, -125, 49), ("2024-07-01", "2024-07-08"))
        assert q.overlaps("1994-02-01", None)

    def test_record_ending_before_the_window_does_not_overlap(self):
        q = Query.from_area((-126, 48, -125, 49), ("2024-07-01", "2024-07-08"))
        assert not q.overlaps("1990-01-01", "1995-01-01")

    def test_record_starting_after_the_window_does_not_overlap(self):
        q = Query.from_area((-126, 48, -125, 49), ("2024-07-01", "2024-07-08"))
        assert not q.overlaps("2025-01-01", None)
