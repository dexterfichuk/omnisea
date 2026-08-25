"""The longer-period ECCC climate collections, against captured real responses.

These four sources exist so a query can reach past the daily record into multi-decadal ones, and
almost everything that can go wrong with them is a labelling mistake rather than a plumbing one:
a month stamped on the wrong day, a mean of daily maxima resampled as a maximum, a station
dropped because the catalogue understates when its record ends, or -9999.9 read as a
temperature. Those are what is tested here.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from conftest import features, load

from omnisea.align import aggregation_for

# Imported as a module so discovery can be stubbed. `omnisea.providers` the *attribute* is a
# function, so pytest's dotted-string setattr target cannot reach the subpackage.
from omnisea.providers import ogc
from omnisea.providers.base import StationMatch
from omnisea.providers.eccc import EcccProvider
from omnisea.providers.eccc.climate import (
    AHCCD_UNIT_FIELDS,
    EcccAhccdAnnual,
    EcccAhccdMonthly,
    EcccAhccdSeasonal,
    EcccClimateMonthly,
)
from omnisea.query import Query

# BLUE RIVER A, 1970 — the one station-year found where every monthly field is populated,
# bright sunshine and snow normals included.
MONTHLY_WINDOW = ("1970-01-01", "1970-04-30")
# AHCCD station 6158355, whose seasonal values were used to pin down the winter convention.
AHCCD_WINDOW = ("1981-01-01", "1981-12-31")
BC_BBOX = (-125.5, 48.0, -122.5, 49.5)


def a_match(source: str, station_id: str = "TEST") -> StationMatch:
    return StationMatch(
        source=source, station_id=station_id, name="Test Station", lat=48.8353, lon=-125.1358
    )


@pytest.fixture
def eccc():
    return EcccProvider()


@pytest.fixture
def monthly_rows():
    return features("climx_monthly.json")


@pytest.fixture
def ahccd_monthly_rows():
    return features("climx_ahccd_monthly.json")


@pytest.fixture
def ahccd_seasonal_rows():
    return features("climx_ahccd_seasonal.json")


@pytest.fixture
def ahccd_annual_rows():
    return features("climx_ahccd_annual.json")


@pytest.fixture
def ahccd_sentinel_rows():
    """ahccd-seasonal for station 7043540, 1979: three seasons missing, one real."""
    return features("climx_ahccd_sentinel.json")


def series(source, rows, window, station="TEST"):
    query = Query.from_area(BC_BBOX, window)
    return source.series_from_rows(query, a_match(source.name, station), rows)


# --------------------------------------------------------------------------- monthly


class TestClimateMonthly:
    def test_month_is_stamped_at_its_first_day(self, eccc, monthly_rows):
        """LOCAL_DATE is "1970-01" — a month with no day, which pandas must not be handed raw."""
        assert monthly_rows[0]["LOCAL_DATE"] == "1970-01"
        frame = series(EcccClimateMonthly(eccc), monthly_rows, MONTHLY_WINDOW).frame
        assert list(frame.index) == [
            pd.Timestamp(f"1970-0{m}-01T00:00:00Z") for m in (1, 2, 3, 4)
        ]

    def test_fields_are_renamed_to_cf(self, eccc, monthly_rows):
        frame = series(EcccClimateMonthly(eccc), monthly_rows, MONTHLY_WINDOW).frame
        assert "air_temperature" in frame.columns
        assert "MEAN_TEMPERATURE" not in frame.columns
        assert "precipitation_amount" in frame.columns
        assert "snowfall_amount" in frame.columns
        assert "duration_of_sunshine" in frame.columns

    def test_normals_are_separate_variables_from_the_observations(self, eccc, monthly_rows):
        """NORMAL_MEAN_TEMPERATURE must never collide with the month actually observed."""
        result = series(EcccClimateMonthly(eccc), monthly_rows, MONTHLY_WINDOW)
        frame = result.frame
        assert {"air_temperature", "air_temperature_normal"} <= set(frame.columns)
        assert frame["air_temperature"].iloc[0] != frame["air_temperature_normal"].iloc[0]
        assert "not an observation" in result.var_attrs["air_temperature_normal"]["comment"].lower()

    def test_normal_repeats_the_same_value_for_a_repeated_month(self, eccc):
        """A normal is a property of the calendar month, so two Januaries must carry one value."""
        rows = features("climx_monthly.json")
        january = dict(rows[0])
        second_january = dict(january, LOCAL_DATE="1971-01")
        frame = series(
            EcccClimateMonthly(eccc), [january, second_january], ("1970-01-01", "1971-12-31")
        ).frame
        assert frame["air_temperature_normal"].nunique() == 1

    def test_extremes_and_totals_carry_the_right_cell_methods(self, eccc, monthly_rows):
        """align() reads these; a total averaged or a maximum interpolated is a wrong number."""
        attrs = series(EcccClimateMonthly(eccc), monthly_rows, MONTHLY_WINDOW).var_attrs
        assert attrs["air_temperature"]["cell_methods"] == "time: mean"
        assert attrs["air_temperature_min"]["cell_methods"] == "time: minimum"
        assert attrs["air_temperature_max"]["cell_methods"] == "time: maximum"
        assert attrs["precipitation_amount"]["cell_methods"] == "time: sum"
        assert attrs["snowfall_amount"]["cell_methods"] == "time: sum"
        assert attrs["heating_degree_days"]["cell_methods"] == "time: sum"

    def test_align_resamples_each_variable_by_its_own_rule(self, eccc, monthly_rows):
        attrs = series(EcccClimateMonthly(eccc), monthly_rows, MONTHLY_WINDOW).var_attrs
        assert aggregation_for(attrs["precipitation_amount"]) == ("sum", "ffill")
        assert aggregation_for(attrs["air_temperature_max"]) == ("max", "ffill")
        assert aggregation_for(attrs["air_temperature_min"]) == ("min", "ffill")
        assert aggregation_for(attrs["air_temperature"]) == ("mean", "ffill")
        # A normal is a mean over years; resampling it must not turn it into a maximum.
        assert aggregation_for(attrs["air_temperature_normal"]) == ("mean", "ffill")
        assert aggregation_for(attrs["precipitation_amount_normal"]) == ("sum", "ffill")

    def test_end_of_month_snow_depth_is_a_point_value(self, eccc, monthly_rows):
        """It is a single reading, not a summary, so align() may interpolate it."""
        attrs = series(EcccClimateMonthly(eccc), monthly_rows, MONTHLY_WINDOW).var_attrs
        snow = attrs["surface_snow_thickness"]
        assert "cell_methods" not in snow
        assert "last day" in snow["comment"]
        assert aggregation_for(snow) == ("mean", "interpolate")

    def test_valid_day_counts_are_kept(self, eccc, monthly_rows):
        """A mean over 12 days is not a mean over 31, and nothing else records the difference."""
        result = series(EcccClimateMonthly(eccc), monthly_rows, MONTHLY_WINDOW)
        assert result.frame["days_with_valid_mean_temperature"].iloc[0] == 31
        # A coverage count has no CF standard name, and inventing one would be worse than
        # leaving it out; the long_name carries the meaning instead.
        attrs = result.var_attrs["days_with_valid_mean_temperature"]
        assert "standard_name" not in attrs
        assert "monthly mean temperature" in attrs["long_name"].lower()

    def test_identity_and_coordinate_columns_do_not_become_variables(self, eccc, monthly_rows):
        """climate-monthly repeats LATITUDE/LONGITUDE; only the geometry may set position."""
        assert "LATITUDE" in monthly_rows[0]
        frame = series(EcccClimateMonthly(eccc), monthly_rows, MONTHLY_WINDOW).frame
        for leaked in ("LATITUDE", "LONGITUDE", "STATION_NAME", "LOCAL_YEAR", "LAST_UPDATED"):
            assert leaked not in frame.columns

    def test_units_stay_provider_units_unless_asked(self, eccc, monthly_rows):
        result = series(EcccClimateMonthly(eccc), monthly_rows, MONTHLY_WINDOW)
        assert result.var_attrs["air_temperature"]["units"] == "degC"
        assert result.frame["air_temperature"].iloc[0] < 100  # still Celsius, not Kelvin

    def test_cf_units_conversion_is_opt_in(self, eccc, monthly_rows):
        source = EcccClimateMonthly(eccc)
        query = Query.from_area(BC_BBOX, MONTHLY_WINDOW, to_cf_units=True)
        result = source.series_from_rows(query, a_match(source.name), monthly_rows)
        assert result.var_attrs["air_temperature"]["units"] == "K"
        assert result.frame["air_temperature"].iloc[0] > 200
        # Bright sunshine is published in hours; CF wants seconds.
        assert result.var_attrs["duration_of_sunshine"]["units"] == "s"

    def test_stale_catalogue_end_date_does_not_reject_a_station(self, eccc):
        """MLY_LAST_DATE understates: 1031316 is listed to 2007 and serves data through 2026."""
        source = EcccClimateMonthly(eccc)
        props = {"MLY_FIRST_DATE": "1970-01-01 00:00:00", "MLY_LAST_DATE": "1979-12-01 00:00:00"}
        first, last = source.record_period(props)
        assert first == "1970-01-01 00:00:00"
        assert last is None
        assert Query.from_area(BC_BBOX, ("2020-01-01", "2021-01-01")).overlaps(first, last)

    def test_a_station_whose_record_starts_later_is_still_rejected(self, eccc):
        source = EcccClimateMonthly(eccc)
        first, last = source.record_period({"MLY_FIRST_DATE": "1990-01-01 00:00:00"})
        assert not Query.from_area(BC_BBOX, ("1970-01-01", "1971-01-01")).overlaps(first, last)

    def test_discover_reads_position_from_geometry(self, eccc, monkeypatch):
        """The station catalogue's LATITUDE is integer micro-degrees and must not be used."""
        stations = load("eccc_stations.json")
        monkeypatch.setattr(ogc, "paginate_ogc_items", lambda *a, **k: iter(stations["features"]))
        source = EcccClimateMonthly(eccc)
        matches = source.discover(Query.from_area((-124.0, 48.0, -123.0, 49.0), MONTHLY_WINDOW))
        assert matches
        for match in matches:
            assert -90 <= match.lat <= 90 and -180 <= match.lon <= 180
            assert match.name

    def test_datetime_param_sends_plain_dates(self, eccc):
        """The collection filters on local dates; a UTC instant is not what it wants."""
        source = EcccClimateMonthly(eccc)
        assert source.datetime_param(Query.from_area(BC_BBOX, MONTHLY_WINDOW)) == (
            "1970-01-01/1970-04-30"
        )

    def test_node_path_does_not_collide_with_the_daily_source(self, eccc):
        assert EcccClimateMonthly(eccc).node_path == "in_situ/weather_monthly"

    def test_time_convention_is_recorded_on_the_node(self, eccc, monthly_rows):
        attrs = series(EcccClimateMonthly(eccc), monthly_rows, MONTHLY_WINDOW).attrs
        assert "first day" in attrs["time_reference"]
        assert attrs["featureType"] == "timeSeries"


# --------------------------------------------------------------------------- AHCCD


class TestAhccdShared:
    def test_missing_marker_is_not_read_as_a_measurement(self, eccc, ahccd_sentinel_rows):
        """-9999.9 degC would sail through every downstream check as a real cold record."""
        raw = [r["temp_mean__temp_moyenne"] for r in ahccd_sentinel_rows]
        assert -9999.9 in raw  # the fixture really does carry the marker

        frame = series(
            EcccAhccdSeasonal(eccc), ahccd_sentinel_rows, ("1979-01-01", "1980-12-31")
        ).frame
        temps = frame["air_temperature"]
        assert temps.min() > -100
        assert temps.count() == 1  # only summer 1979 was measured
        assert temps.dropna().iloc[0] == pytest.approx(17.0)

    def test_a_column_that_is_entirely_missing_is_dropped(self, eccc, ahccd_sentinel_rows):
        """Every pressure value in this fixture is null; an all-NaN column is noise."""
        frame = series(
            EcccAhccdSeasonal(eccc), ahccd_sentinel_rows, ("1979-01-01", "1980-12-31")
        ).frame
        assert "air_pressure_at_sea_level" not in frame.columns

    def test_partly_missing_columns_keep_their_real_values(self, eccc, ahccd_sentinel_rows):
        result = series(
            EcccAhccdSeasonal(eccc), ahccd_sentinel_rows, ("1979-01-01", "1980-12-31")
        )
        rain = result.frame["rainfall_amount"]
        assert rain.count() == 1
        assert rain.dropna().iloc[0] == pytest.approx(259.2)

    def test_units_come_from_the_sibling_property(self, eccc, ahccd_monthly_rows):
        """AHCCD publishes units per record rather than fixing them in collection metadata."""
        result = series(EcccAhccdMonthly(eccc), ahccd_monthly_rows, AHCCD_WINDOW)
        assert result.var_attrs["air_temperature"]["units"] == "C"

        rainy = series(
            EcccAhccdSeasonal(eccc),
            features("climx_ahccd_sentinel.json"),
            ("1979-01-01", "1980-12-31"),
        )
        assert rainy.var_attrs["precipitation_amount"]["units"] == "mm"
        assert rainy.var_attrs["rainfall_amount"]["units"] == "mm"

    def test_units_properties_do_not_become_variables(self, eccc, ahccd_monthly_rows):
        assert "temp_mean_units__temp_moyenne_unites" in ahccd_monthly_rows[0]
        frame = series(EcccAhccdMonthly(eccc), ahccd_monthly_rows, AHCCD_WINDOW).frame
        for column in frame.columns:
            assert "_units__" not in column

    def test_every_units_sibling_is_declared(self, eccc, ahccd_monthly_rows):
        """A units field omitted from the map would leak through as a string variable."""
        published = {k for k in ahccd_monthly_rows[0] if "_units__" in k}
        assert published == set(AHCCD_UNIT_FIELDS.values())

    def test_mean_of_daily_maxima_resamples_as_a_mean(self, eccc, ahccd_monthly_rows):
        """temp_max is the month's mean daily maximum, not its highest reading.

        Verified upstream: temp_mean is the midpoint of temp_max and temp_min to within
        0.2 degC. Spelling this "time: maximum" would have align() take a maximum over a series
        of means when downsampling.
        """
        result = series(EcccAhccdMonthly(eccc), ahccd_monthly_rows, AHCCD_WINDOW)
        attrs = result.var_attrs["air_temperature_max"]
        assert attrs["cell_methods"] == "time: mean"
        assert aggregation_for(attrs) == ("mean", "ffill")
        assert "mean daily maximum" in attrs["long_name"].lower()

    def test_snowfall_is_water_equivalent_not_depth(self, eccc, ahccd_sentinel_rows):
        """rain + snow equals total_precip exactly, which a depth in cm could not do."""
        row = next(r for r in ahccd_sentinel_rows if r["period_value__valeur_periode"] == "Fal")
        assert row["rain__pluie"] + row["snow__neige"] == pytest.approx(
            row["total_precip__precip_totale"]
        )
        result = series(
            EcccAhccdSeasonal(eccc), ahccd_sentinel_rows, ("1979-01-01", "1980-12-31")
        )
        assert (
            result.var_attrs["snowfall_amount"]["standard_name"]
            == "lwe_thickness_of_snowfall_amount"
        )

    def test_station_name_comes_from_the_ahccd_field(self, eccc, monkeypatch):
        """ahccd-stations names the station bilingually, not in STATION_NAME."""
        stations = load("climx_ahccd_stations.json")
        assert "STATION_NAME" not in stations["features"][0]["properties"]
        monkeypatch.setattr(ogc, "paginate_ogc_items", lambda *a, **k: iter(stations["features"]))
        matches = EcccAhccdMonthly(eccc).discover(
            Query.from_area(BC_BBOX, ("1960-01-01", "1990-01-01"))
        )
        assert matches
        assert all(m.name for m in matches)
        assert any("BURQUITLAM" in m.name for m in matches)

    def test_stale_catalogue_end_date_does_not_reject_a_station(self, eccc, monkeypatch):
        """1171020 is listed as ending 2004-03 and has monthly data through 2017-12."""
        stations = load("climx_ahccd_stations.json")
        ends = [f["properties"]["end_date__date_fin"] for f in stations["features"]]
        assert min(ends) < "2006"  # 1101200, listed as ending 2005-12
        monkeypatch.setattr(ogc, "paginate_ogc_items", lambda *a, **k: iter(stations["features"]))
        matches = EcccAhccdMonthly(eccc).discover(
            Query.from_area(BC_BBOX, ("2016-01-01", "2017-12-31"))
        )
        assert matches

    def test_station_starting_after_the_window_is_still_rejected(self, eccc):
        source = EcccAhccdMonthly(eccc)
        first, last = source.record_period({"start_date__date_debut": "1990-01-01"})
        assert last is None
        assert not Query.from_area(BC_BBOX, ("1900-01-01", "1910-01-01")).overlaps(first, last)

    def test_node_paths_are_distinct(self, eccc):
        paths = {
            source(eccc).node_path
            for source in (EcccAhccdMonthly, EcccAhccdSeasonal, EcccAhccdAnnual)
        }
        assert paths == {"in_situ/ahccd_monthly", "in_situ/ahccd_seasonal", "in_situ/ahccd_annual"}
        assert EcccClimateMonthly(eccc).node_path not in paths

    def test_adjustment_is_declared_on_the_node(self, eccc, ahccd_monthly_rows):
        attrs = series(EcccAhccdMonthly(eccc), ahccd_monthly_rows, AHCCD_WINDOW).attrs
        assert "homogenized" in attrs["comment"].lower()


class TestAhccdMonthlyTime:
    def test_month_is_stamped_at_its_first_day(self, eccc, ahccd_monthly_rows):
        assert ahccd_monthly_rows[0]["date"] == "1981-04"
        frame = series(EcccAhccdMonthly(eccc), ahccd_monthly_rows, AHCCD_WINDOW).frame
        assert list(frame.index) == [
            pd.Timestamp(f"1981-0{m}-01T00:00:00Z") for m in (1, 2, 3, 4)
        ]

    def test_rows_arrive_unsorted_and_come_back_ordered(self, eccc, ahccd_monthly_rows):
        """The collection does not return months in order."""
        assert [r["date"] for r in ahccd_monthly_rows] != sorted(
            r["date"] for r in ahccd_monthly_rows
        )
        frame = series(EcccAhccdMonthly(eccc), ahccd_monthly_rows, AHCCD_WINDOW).frame
        assert frame.index.is_monotonic_increasing


class TestAhccdSeasonalTime:
    def test_winter_is_stamped_at_the_previous_december(self, eccc, ahccd_seasonal_rows):
        """Winter 1981 is December 1980 through February 1981, confirmed against the months."""
        winter = next(r for r in ahccd_seasonal_rows if r["period_value__valeur_periode"] == "Win")
        assert winter["year__annee"] == 1981
        assert EcccAhccdSeasonal(eccc).extract_time(winter) == "1980-12-01T00:00:00Z"

    def test_each_season_is_stamped_at_the_month_it_begins_in(self, eccc, ahccd_seasonal_rows):
        source = EcccAhccdSeasonal(eccc)
        stamped = {
            r["period_value__valeur_periode"]: source.extract_time(r) for r in ahccd_seasonal_rows
        }
        assert stamped == {
            "Win": "1980-12-01T00:00:00Z",
            "Spr": "1981-03-01T00:00:00Z",
            "Smr": "1981-06-01T00:00:00Z",
            "Fal": "1981-09-01T00:00:00Z",
        }

    def test_the_winter_a_window_starts_before_is_trimmed_away(self, eccc, ahccd_seasonal_rows):
        """Winter 1981 begins in December 1980, outside a window opening on 1 January 1981."""
        frame = series(EcccAhccdSeasonal(eccc), ahccd_seasonal_rows, AHCCD_WINDOW).frame
        assert pd.Timestamp("1980-12-01T00:00:00Z") not in frame.index
        assert len(frame) == 3

    def test_the_request_is_padded_so_a_december_winter_is_not_missed(self, eccc):
        """Upstream filters on the season's year, which is not where omnisea stamps it."""
        source = EcccAhccdSeasonal(eccc)
        param = source.datetime_param(Query.from_area(BC_BBOX, ("1981-01-01", "1990-12-31")))
        start, end = param.split("/")
        assert start == "1980-01-01"
        assert end == "1991-12-31"  # so winter 1991, which begins 1990-12-01, is returned

    def test_an_unknown_season_label_is_dropped_not_guessed(self, eccc):
        source = EcccAhccdSeasonal(eccc)
        unknown = {"year__annee": 1981, "period_value__valeur_periode": "??"}
        assert source.extract_time(unknown) is None
        assert source.extract_time({"period_value__valeur_periode": "Smr"}) is None


class TestAhccdAnnualTime:
    def test_year_is_stamped_at_the_first_of_january(self, eccc, ahccd_annual_rows):
        frame = series(EcccAhccdAnnual(eccc), ahccd_annual_rows, ("1979-01-01", "1981-12-31")).frame
        assert list(frame.index) == [
            pd.Timestamp(f"{y}-01-01T00:00:00Z") for y in (1979, 1980, 1981)
        ]

    def test_a_row_with_no_year_is_dropped(self, eccc):
        source = EcccAhccdAnnual(eccc)
        assert source.extract_time({"year__annee": None}) is None
        assert source.extract_time({"year__annee": "not a year"}) is None

    def test_row_estimate_reads_as_years(self, eccc):
        source = EcccAhccdAnnual(eccc)
        query = Query.from_area(BC_BBOX, ("1900-01-01", "2000-01-01"))
        assert math.isclose(query.days * source.samples_per_day, 100, abs_tol=1)


# --------------------------------------------------------------------------- live


@pytest.mark.network
class TestLiveCollections:
    """Real requests. These catch what a fixture cannot: an upstream shape change."""

    CAPE_BEALE = "1031316"

    def test_climate_monthly_returns_decades_of_real_months(self, eccc):
        source = EcccClimateMonthly(eccc)
        query = Query.from_area((-125.4, 48.6, -125.0, 49.0), ("1990-01-01", "1999-12-31"))
        match = source.new_match(
            station_id=self.CAPE_BEALE, name="CAPE BEALE LIGHT", lat=48.786, lon=-125.216
        )
        result = source.fetch_station(query, match)
        assert result is not None
        frame = result.frame
        assert len(frame) == 120  # ten complete years, one row per month
        assert frame.index.is_monotonic_increasing
        assert frame.index.min() == pd.Timestamp("1990-01-01T00:00:00Z")
        assert frame.index.max() == pd.Timestamp("1999-12-01T00:00:00Z")
        assert frame["air_temperature"].between(-10, 30).all()
        assert result.var_attrs["air_temperature"]["cell_methods"] == "time: mean"

    def test_climate_monthly_serves_a_station_its_catalogue_calls_finished(self, eccc):
        """1031316 is listed as ending 2007-02; the collection has it through last year."""
        source = EcccClimateMonthly(eccc)
        query = Query.from_area((-125.4, 48.6, -125.0, 49.0), ("2020-01-01", "2020-12-31"))
        match = source.new_match(
            station_id=self.CAPE_BEALE, name="CAPE BEALE LIGHT", lat=48.786, lon=-125.216
        )
        result = source.fetch_station(query, match)
        assert result is not None and not result.frame.empty

    def test_ahccd_monthly_returns_adjusted_values(self, eccc):
        source = EcccAhccdMonthly(eccc)
        query = Query.from_area((-80.0, 43.0, -79.0, 44.0), ("1981-01-01", "1981-12-31"))
        match = source.new_match(station_id="6158355", name="TORONTO", lat=43.67, lon=-79.4)
        result = source.fetch_station(query, match)
        assert result is not None
        assert len(result.frame) == 12
        temps = result.frame["air_temperature"]
        assert temps.min() > -60  # the -9999.9 marker never reaches the frame
        assert result.var_attrs["air_temperature"]["units"] == "C"

    def test_ahccd_seasonal_winter_matches_the_months_it_spans(self, eccc):
        """Winter 1981 must equal the mean of December 1980, January and February 1981."""
        monthly = EcccAhccdMonthly(eccc)
        seasonal = EcccAhccdSeasonal(eccc)
        area = (-80.0, 43.0, -79.0, 44.0)
        match_for = lambda s: s.new_match(  # noqa: E731 - one-liner, three uses
            station_id="6158355", name="TORONTO", lat=43.67, lon=-79.4
        )
        months = monthly.fetch_station(
            Query.from_area(area, ("1980-12-01", "1981-02-28")), match_for(monthly)
        ).frame["air_temperature"]
        assert len(months) == 3
        seasons = seasonal.fetch_station(
            Query.from_area(area, ("1980-12-01", "1981-11-30")), match_for(seasonal)
        ).frame
        winter = seasons.loc[pd.Timestamp("1980-12-01T00:00:00Z"), "air_temperature"]
        assert winter == pytest.approx(months.mean(), abs=0.15)

    def test_ahccd_annual_reaches_back_further_than_the_daily_record(self, eccc):
        source = EcccAhccdAnnual(eccc)
        query = Query.from_area((-80.0, 43.0, -79.0, 44.0), ("1900-01-01", "1999-12-31"))
        match = source.new_match(station_id="6158355", name="TORONTO", lat=43.67, lon=-79.4)
        result = source.fetch_station(query, match)
        assert result is not None
        assert len(result.frame) > 50
        assert result.frame.index.min() < pd.Timestamp("1950-01-01T00:00:00Z")

    def test_ahccd_stations_discovery_names_its_stations(self, eccc):
        matches = EcccAhccdMonthly(eccc).discover(
            Query.from_area(BC_BBOX, ("1960-01-01", "1990-01-01"))
        )
        assert matches
        assert all(m.name for m in matches)
        assert all(-90 <= m.lat <= 90 for m in matches)

    def test_climate_normals_has_no_time_axis(self, eccc):
        """Why climate-normals is not wired as a source: a datetime filter matches nothing.

        The shared fetch path always sends one, so every query would come back empty. The
        collection is a station x element x month table, not a series.
        """
        from omnisea.http import get_json

        url = eccc.collection_url("climate-normals")
        params = {"CLIMATE_IDENTIFIER": self.CAPE_BEALE, "limit": 1, "f": "json"}
        unfiltered = get_json(url, params, provider="eccc")
        assert unfiltered["numberMatched"] > 500

        filtered = get_json(
            url, dict(params, datetime="1990-01-01/2000-01-01"), provider="eccc"
        )
        assert filtered["numberMatched"] == 0

        props = unfiltered["features"][0]["properties"]
        # One prose-named element and one VALUE per (station, element, month) — 66 elements
        # against MONTH 1-12 plus 13 for the annual figure.
        assert {"MONTH", "VALUE", "E_NORMAL_ELEMENT_NAME"} <= set(props)
        assert props["PERIOD_BEGIN"] == 1981 and props["PERIOD_END"] == 2010
        # None of the observation-time fields the wired sources key on are present. The only
        # dates here describe when the normal was computed, not when anything was measured.
        assert not ({"LOCAL_DATE", "UTC_DATE", "date", "year__annee"} & set(props))
