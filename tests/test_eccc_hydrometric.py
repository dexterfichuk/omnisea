"""The five ECCC hydrometric collections, shaped offline against captured API responses.

``hydrometric-realtime`` holds about 30 days, so everything here about *historical* river data
exercises the four HYDAT collections behind it. The awkward parts of those — a year that exists
only inside an IDENTIFIER, peak times published in local standard time, and annual rows that
collide on time unless they are widened — are what most of these tests are about.
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import features, load

from omnisea.providers.base import StationMatch
from omnisea.providers.eccc import EcccProvider
from omnisea.providers.eccc.hydrometric import (
    DISCHARGE_CF,
    LEVEL_CF,
    EcccHydrometric,
    EcccHydrometricAnnualPeaks,
    EcccHydrometricAnnualStatistics,
    EcccHydrometricDailyMean,
    EcccHydrometricMonthlyMean,
)
from omnisea.providers.ogc import point_from_feature
from omnisea.query import Query

# Sarita River near Bamfield — the running example, and an active gauge with a long HYDAT record.
SARITA = "08HB014"
SARITA_LAT, SARITA_LON = 48.8925, -124.9694
# Nahmint River near Port Alberni: discontinued in the catalogue, decades of record in HYDAT.
NAHMINT = "08HB012"

BARKLEY_SOUND = (-125.5, 48.5, -124.5, 49.2)
WEEK_2024 = ("2024-07-01", "2024-07-08")
YEAR_2020 = ("2020-01-01", "2020-12-31")
YEARS_2018_2020 = ("2018-01-01", "2020-12-31")

HISTORICAL = (
    EcccHydrometricDailyMean,
    EcccHydrometricMonthlyMean,
    EcccHydrometricAnnualStatistics,
    EcccHydrometricAnnualPeaks,
)


@pytest.fixture
def eccc():
    return EcccProvider()


@pytest.fixture
def daily_rows():
    return features("hydro_daily_mean.json")


@pytest.fixture
def monthly_rows():
    return features("hydro_monthly_mean.json")


@pytest.fixture
def annual_rows():
    return features("hydro_annual_statistics.json")


@pytest.fixture
def annual_gap_rows():
    """02OJ034: three years whose MIN_DATE is null, so the year cannot come from the dates."""
    return features("hydro_annual_statistics_gaps.json")


@pytest.fixture
def peak_rows():
    return features("hydro_annual_peaks.json")


@pytest.fixture
def peak_gap_rows():
    """01AF009: the one row in ~2000 with no DATE and no PEAK at all."""
    return features("hydro_annual_peaks_gaps.json")


@pytest.fixture
def hydro_stations():
    """The seven gauges around Barkley Sound: two active, five discontinued."""
    return load("hydro_stations.json")


def a_match(station_id: str = SARITA) -> StationMatch:
    return StationMatch(
        source="test",
        station_id=station_id,
        name="SARITA RIVER NEAR BAMFIELD",
        lat=SARITA_LAT,
        lon=SARITA_LON,
    )


def shaped(source, rows, window):
    """Run one source's shaping path and hand back the series it produces."""
    return source.series_from_rows(Query.from_area(BARKLEY_SOUND, window), a_match(), rows)


# --------------------------------------------------------------------------- daily mean


class TestDailyMean:
    def test_a_2024_week_comes_back_instead_of_nothing(self, eccc, daily_rows):
        """The gap this source exists to close: realtime holds ~30 days, HYDAT holds 2024."""
        series = shaped(EcccHydrometricDailyMean(eccc), daily_rows, WEEK_2024)
        assert len(series.frame) == 8
        assert series.frame.index[0] == pd.Timestamp("2024-07-01T00:00:00Z")
        assert series.frame.index[-1] == pd.Timestamp("2024-07-08T00:00:00Z")

    def test_level_and_discharge_get_cf_names(self, eccc, daily_rows):
        series = shaped(EcccHydrometricDailyMean(eccc), daily_rows, WEEK_2024)
        assert series.frame[LEVEL_CF].iloc[0] == pytest.approx(1.502, abs=1e-3)
        assert series.frame[DISCHARGE_CF].iloc[0] == pytest.approx(2.57, abs=1e-2)

    def test_daily_means_are_marked_as_means(self, eccc, daily_rows):
        """align() reads this to decide that a daily mean is averaged, never interpolated."""
        series = shaped(EcccHydrometricDailyMean(eccc), daily_rows, WEEK_2024)
        assert series.var_attrs[LEVEL_CF]["cell_methods"] == "time: mean"
        assert series.var_attrs[DISCHARGE_CF]["cell_methods"] == "time: mean"

    def test_the_time_field_is_date_not_datetime(self, eccc):
        """hydrometric-daily-mean publishes DATE; only the realtime collection has DATETIME."""
        source = EcccHydrometricDailyMean(eccc)
        assert source.time_field == "DATE"
        assert EcccHydrometric(eccc).time_field == "DATETIME"

    def test_a_calendar_date_is_stamped_at_midnight_utc(self, eccc):
        source = EcccHydrometricDailyMean(eccc)
        assert source.extract_time({"DATE": "2024-07-01"}) == "2024-07-01T00:00:00Z"

    def test_bilingual_symbol_columns_do_not_become_variables(self, eccc, daily_rows):
        series = shaped(EcccHydrometricDailyMean(eccc), daily_rows, WEEK_2024)
        assert not [c for c in series.frame.columns if "SYMBOL" in str(c)]

    def test_datetime_param_uses_plain_dates(self, eccc):
        source = EcccHydrometricDailyMean(eccc)
        param = source.datetime_param(Query.from_area(BARKLEY_SOUND, WEEK_2024))
        assert param == "2024-07-01/2024-07-08"

    def test_nodes_do_not_collide_with_the_realtime_source(self, eccc):
        paths = {cls(eccc).node_path for cls in (*HISTORICAL, EcccHydrometric)}
        assert len(paths) == 5


# --------------------------------------------------------------------------- monthly mean


class TestMonthlyMean:
    def test_a_year_month_with_no_day_becomes_the_first_of_that_month(self, eccc):
        """DATE here is "2020-01" — a period, not a date, and not parseable as an instant."""
        source = EcccHydrometricMonthlyMean(eccc)
        assert source.extract_time({"DATE": "2020-01"}) == "2020-01-01T00:00:00Z"

    def test_monthly_rows_land_on_month_starts(self, eccc, monthly_rows):
        series = shaped(EcccHydrometricMonthlyMean(eccc), monthly_rows, YEAR_2020)
        assert list(series.frame.index[:3]) == [
            pd.Timestamp("2020-01-01T00:00:00Z"),
            pd.Timestamp("2020-02-01T00:00:00Z"),
            pd.Timestamp("2020-03-01T00:00:00Z"),
        ]
        assert series.frame[DISCHARGE_CF].iloc[0] == pytest.approx(72.2, abs=1e-1)

    def test_monthly_means_are_marked_as_means(self, eccc, monthly_rows):
        series = shaped(EcccHydrometricMonthlyMean(eccc), monthly_rows, YEAR_2020)
        assert series.var_attrs[LEVEL_CF]["cell_methods"] == "time: mean"

    def test_the_prefixed_upstream_names_are_mapped_not_carried(self, eccc, monthly_rows):
        """MONTHLY_MEAN_LEVEL must become the CF name, not survive as a passthrough column."""
        series = shaped(EcccHydrometricMonthlyMean(eccc), monthly_rows, YEAR_2020)
        assert "MONTHLY_MEAN_LEVEL" not in series.frame.columns
        assert LEVEL_CF in series.frame.columns


# --------------------------------------------------------------------------- period windows


class TestPeriodWindows:
    """An aggregate is labelled by the start of the period it covers, at both ends of the trip.

    Upstream matches on the covered period, so a window landing inside one matches nothing at
    all — verified live, ``hydrometric-monthly-mean`` gives 0 rows for 2020-07-15/2020-07-20 and
    1 row for the whole month.
    """

    def test_a_mid_month_window_is_grown_to_the_whole_month(self, eccc):
        source = EcccHydrometricMonthlyMean(eccc)
        query = Query.from_area(BARKLEY_SOUND, ("2020-07-15", "2020-07-20"))
        assert source.datetime_param(query) == "2020-07-01/2020-07-31"

    def test_a_mid_year_window_is_grown_to_the_whole_year(self, eccc):
        source = EcccHydrometricAnnualStatistics(eccc)
        query = Query.from_area(BARKLEY_SOUND, ("2020-06-01", "2020-09-30"))
        assert source.datetime_param(query) == "2020-01-01/2020-12-31"

    def test_a_window_spanning_years_is_grown_at_both_ends(self, eccc):
        source = EcccHydrometricAnnualPeaks(eccc)
        query = Query.from_area(BARKLEY_SOUND, ("2018-06-01", "2020-09-30"))
        assert source.datetime_param(query) == "2018-01-01/2020-12-31"

    def test_a_daily_window_is_unchanged(self, eccc):
        source = EcccHydrometricDailyMean(eccc)
        query = Query.from_area(BARKLEY_SOUND, WEEK_2024)
        assert source.datetime_param(query) == "2024-07-01/2024-07-08"

    def test_the_year_row_survives_a_mid_year_request(self, eccc, annual_rows):
        """Without the widening the trim drops it: the row is stamped 1 January."""
        series = shaped(
            EcccHydrometricAnnualStatistics(eccc), annual_rows, ("2020-06-01", "2020-09-30")
        )
        assert list(series.frame.index) == [pd.Timestamp("2020-01-01T00:00:00Z")]

    def test_the_month_row_survives_a_mid_month_request(self, eccc, monthly_rows):
        series = shaped(
            EcccHydrometricMonthlyMean(eccc), monthly_rows, ("2020-03-10", "2020-03-20")
        )
        assert list(series.frame.index) == [pd.Timestamp("2020-03-01T00:00:00Z")]

    def test_rows_outside_the_grown_window_are_still_trimmed(self, eccc, monthly_rows):
        """Growing the window must not turn the trim off altogether."""
        series = shaped(
            EcccHydrometricMonthlyMean(eccc), monthly_rows, ("2020-02-10", "2020-02-20")
        )
        assert list(series.frame.index) == [pd.Timestamp("2020-02-01T00:00:00Z")]


# --------------------------------------------------------------------------- annual statistics


class TestAnnualStatistics:
    def test_level_and_discharge_both_survive_the_pivot(self, eccc, annual_rows):
        """Left long these rows collide on time and one quantity is silently dropped."""
        series = shaped(EcccHydrometricAnnualStatistics(eccc), annual_rows, YEARS_2018_2020)
        row = series.frame.loc[pd.Timestamp("2018-01-01T00:00:00Z")]
        assert row[f"{LEVEL_CF}_max"] == pytest.approx(4.154, abs=1e-3)
        assert row[f"{DISCHARGE_CF}_max"] == pytest.approx(317.0, abs=1e-1)

    def test_one_row_per_year(self, eccc, annual_rows):
        series = shaped(EcccHydrometricAnnualStatistics(eccc), annual_rows, YEARS_2018_2020)
        assert list(series.frame.index) == [
            pd.Timestamp(f"{year}-01-01T00:00:00Z") for year in (2018, 2019, 2020)
        ]

    def test_extremes_carry_maximum_and_minimum_cell_methods(self, eccc, annual_rows):
        """An annual maximum resampled as a mean is not a statistic of anything."""
        series = shaped(EcccHydrometricAnnualStatistics(eccc), annual_rows, YEARS_2018_2020)
        assert series.var_attrs[f"{LEVEL_CF}_max"]["cell_methods"] == "time: maximum"
        assert series.var_attrs[f"{LEVEL_CF}_min"]["cell_methods"] == "time: minimum"
        assert series.var_attrs[f"{DISCHARGE_CF}_max"]["cell_methods"] == "time: maximum"
        assert series.var_attrs[f"{DISCHARGE_CF}_min"]["cell_methods"] == "time: minimum"

    def test_the_date_each_extreme_fell_on_is_kept(self, eccc, annual_rows):
        series = shaped(EcccHydrometricAnnualStatistics(eccc), annual_rows, YEARS_2018_2020)
        row = series.frame.loc[pd.Timestamp("2018-01-01T00:00:00Z")]
        assert row[f"{LEVEL_CF}_max_time"] == "2018-01-21"
        assert row[f"{LEVEL_CF}_min_time"] == "2018-09-06"

    def test_a_flag_present_on_one_quantity_attaches_to_that_quantity(self, eccc, annual_rows):
        series = shaped(EcccHydrometricAnnualStatistics(eccc), annual_rows, YEARS_2018_2020)
        assert series.frame[f"{DISCHARGE_CF}_max_qc"].iloc[0] == "Estimated"

    def test_empty_string_flags_do_not_become_a_column_of_blanks(self, eccc, annual_rows):
        """This collection writes "no flag" as "", where every other one writes null."""
        series = shaped(EcccHydrometricAnnualStatistics(eccc), annual_rows, YEARS_2018_2020)
        assert f"{LEVEL_CF}_max_qc" not in series.frame.columns
        for column in series.frame.columns:
            assert not (series.frame[column] == "").any()

    def test_the_year_is_read_when_both_dates_are_null(self, eccc, annual_gap_rows):
        """There is no year field; MIN_DATE is null here, so it comes from the IDENTIFIER."""
        series = shaped(
            EcccHydrometricAnnualStatistics(eccc), annual_gap_rows, ("1996-01-01", "1998-12-31")
        )
        assert list(series.frame.index) == [
            pd.Timestamp(f"{year}-01-01T00:00:00Z") for year in (1996, 1997, 1998)
        ]

    def test_a_null_extreme_stays_missing(self, eccc, annual_gap_rows):
        series = shaped(
            EcccHydrometricAnnualStatistics(eccc), annual_gap_rows, ("1996-01-01", "1998-12-31")
        )
        row = series.frame.loc[pd.Timestamp("1996-01-01T00:00:00Z")]
        assert pd.isna(row[f"{LEVEL_CF}_min"])
        assert row[f"{LEVEL_CF}_max"] == pytest.approx(6.99, abs=1e-2)


# --------------------------------------------------------------------------- annual peaks


class TestAnnualPeaks:
    def test_peak_times_are_converted_from_local_standard_time(self, eccc, peak_rows):
        """DATE is local with the offset in TIMEZONE_OFFSET; 09:00 at -8 is 17:00Z."""
        series = shaped(EcccHydrometricAnnualPeaks(eccc), peak_rows, YEARS_2018_2020)
        row = series.frame.loc[pd.Timestamp("2018-01-01T00:00:00Z")]
        assert row[f"{LEVEL_CF}_max_time"] == "2018-01-21T17:00:00Z"

    def test_a_peak_pushed_into_the_next_utc_month_keeps_its_own_year(self, eccc, peak_rows):
        """The 2020 maximum is 31 January local, 1 February in UTC; the row is still 2020."""
        series = shaped(EcccHydrometricAnnualPeaks(eccc), peak_rows, YEARS_2018_2020)
        row = series.frame.loc[pd.Timestamp("2020-01-01T00:00:00Z")]
        assert row[f"{LEVEL_CF}_max_time"] == "2020-02-01T04:40:00Z"

    def test_both_quantities_survive_a_shared_peak_instant(self, eccc, peak_rows):
        """The 2018 level and discharge maxima are the same flood, stamped the same minute."""
        series = shaped(EcccHydrometricAnnualPeaks(eccc), peak_rows, YEARS_2018_2020)
        row = series.frame.loc[pd.Timestamp("2018-01-01T00:00:00Z")]
        assert row[f"{LEVEL_CF}_max"] == pytest.approx(5.129, abs=1e-3)
        assert row[f"{DISCHARGE_CF}_max"] == pytest.approx(524.0, abs=1e-1)
        assert row[f"{LEVEL_CF}_max_time"] == row[f"{DISCHARGE_CF}_max_time"]

    def test_an_instantaneous_peak_exceeds_the_daily_mean_extreme(
        self, eccc, peak_rows, annual_rows
    ):
        """A sanity check across the two annual collections: 5.129 m peak, 4.154 m daily mean."""
        peaks = shaped(EcccHydrometricAnnualPeaks(eccc), peak_rows, YEARS_2018_2020).frame
        stats = shaped(
            EcccHydrometricAnnualStatistics(eccc), annual_rows, YEARS_2018_2020
        ).frame
        at = pd.Timestamp("2018-01-01T00:00:00Z")
        assert peaks.loc[at, f"{LEVEL_CF}_max"] > stats.loc[at, f"{LEVEL_CF}_max"]
        assert peaks.loc[at, f"{DISCHARGE_CF}_max"] > stats.loc[at, f"{DISCHARGE_CF}_max"]

    def test_peaks_carry_maximum_and_minimum_cell_methods(self, eccc, peak_rows):
        series = shaped(EcccHydrometricAnnualPeaks(eccc), peak_rows, YEARS_2018_2020)
        assert series.var_attrs[f"{DISCHARGE_CF}_max"]["cell_methods"] == "time: maximum"
        assert series.var_attrs[f"{DISCHARGE_CF}_min"]["cell_methods"] == "time: minimum"

    def test_a_row_with_no_date_at_all_still_lands_on_its_year(self, eccc, peak_gap_rows):
        """One row in ~2000 has a null DATE and a null PEAK; the IDENTIFIER still names 1991."""
        series = shaped(
            EcccHydrometricAnnualPeaks(eccc), peak_gap_rows, ("1991-01-01", "1993-12-31")
        )
        assert pd.Timestamp("1991-01-01T00:00:00Z") in series.frame.index

    def test_the_units_prose_is_not_mistaken_for_a_unit(self, eccc, peak_rows):
        """UNITS_EN reads "in metres (to millimetres)" and is null for discharge."""
        series = shaped(EcccHydrometricAnnualPeaks(eccc), peak_rows, YEARS_2018_2020)
        assert series.var_attrs[f"{LEVEL_CF}_max"]["units"] == "m"
        assert series.var_attrs[f"{DISCHARGE_CF}_max"]["units"] == "m3 s-1"
        assert "UNITS_EN" not in series.frame.columns


# --------------------------------------------------------------------------- discovery


class TestHistoricalDiscovery:
    def test_discontinued_gauges_are_kept(self, eccc, hydro_stations):
        """HYDAT is mostly discontinued gauges; only the realtime source may reject them."""
        query = Query.from_area(BARKLEY_SOUND, WEEK_2024)
        source = EcccHydrometricDailyMean(eccc)
        found = {
            match.station_id
            for feature in hydro_stations["features"]
            if (match := source.station_from_feature(query, feature)) is not None
        }
        assert NAHMINT in found
        assert SARITA in found

    def test_the_realtime_source_rejects_the_same_discontinued_gauges(
        self, eccc, hydro_stations
    ):
        query = Query.from_area(BARKLEY_SOUND, WEEK_2024)
        source = EcccHydrometric(eccc)
        found = {
            match.station_id
            for feature in hydro_stations["features"]
            if (match := source.station_from_feature(query, feature)) is not None
        }
        assert NAHMINT not in found
        assert SARITA in found

    def test_coordinates_come_from_geometry(self, eccc, hydro_stations):
        feature = next(
            f for f in hydro_stations["features"] if f["properties"]["STATION_NUMBER"] == SARITA
        )
        lat, lon = point_from_feature(feature)
        assert lat == pytest.approx(SARITA_LAT, abs=1e-3)
        assert lon == pytest.approx(SARITA_LON, abs=1e-3)

    def test_the_vertical_datum_reaches_the_node(self, eccc, hydro_stations):
        """A water level without its datum is a number, not a measurement."""
        query = Query.from_area(BARKLEY_SOUND, WEEK_2024)
        source = EcccHydrometricDailyMean(eccc)
        feature = next(
            f for f in hydro_stations["features"] if f["properties"]["STATION_NUMBER"] == SARITA
        )
        match = source.station_from_feature(query, feature)
        assert source.node_attrs(query, match)["datum"] == "ASSUMED DATUM"

    def test_the_time_convention_is_recorded_on_every_node(self, eccc):
        query = Query.from_area(BARKLEY_SOUND, WEEK_2024)
        for cls in HISTORICAL:
            source = cls(eccc)
            attrs = source.node_attrs(query, a_match())
            assert attrs["time_reference"], f"{cls.__name__} states no time convention"


# --------------------------------------------------------------------------- conventions


class TestConventions:
    def test_every_standard_name_is_one_of_the_two_real_cf_names(self, eccc):
        """Checked against the CF standard name table; a timestamp gets no standard name."""
        for cls in (*HISTORICAL, EcccHydrometric):
            for raw, spec in cls(eccc).fields.items():
                assert spec.standard_name in ("", LEVEL_CF, DISCHARGE_CF), (
                    f"{cls.__name__}.{raw} claims standard_name {spec.standard_name!r}"
                )

    def test_variables_selects_stations_not_columns(self, eccc, daily_rows):
        """`variables=` must never project the response; the columns already crossed the wire."""
        source = EcccHydrometricDailyMean(eccc)
        query = Query.from_area(BARKLEY_SOUND, WEEK_2024, variables=[LEVEL_CF])
        series = source.series_from_rows(query, a_match(), daily_rows)
        assert DISCHARGE_CF in series.frame.columns

    def test_each_source_advertises_both_quantities(self, eccc):
        for cls in HISTORICAL:
            assert cls(eccc).variables >= {LEVEL_CF, DISCHARGE_CF}, cls.__name__

    def test_row_estimates_count_periods_and_never_round_to_zero(self, eccc, hydro_stations):
        """A year of annual peaks is one row — and "~0 rows" would read as "nothing here"."""
        query = Query.from_area(BARKLEY_SOUND, YEAR_2020)
        feature = next(
            f for f in hydro_stations["features"] if f["properties"]["STATION_NUMBER"] == SARITA
        )
        estimates = [cls(eccc).station_from_feature(query, feature).n_rows_est
                     for cls in HISTORICAL]
        assert estimates == [366, 12, 1, 1]  # 2020 is a leap year

    def test_names_and_collections_are_distinct(self, eccc):
        sources = [cls(eccc) for cls in (*HISTORICAL, EcccHydrometric)]
        assert len({s.name for s in sources}) == 5
        assert len({s.collection for s in sources}) == 5


# --------------------------------------------------------------------------- live


@pytest.mark.network
class TestHydrometricLive:
    """Proof that the historical collections answer for a year realtime cannot reach."""

    def test_daily_mean_returns_real_2024_bamfield_river_data(self, eccc):
        source = EcccHydrometricDailyMean(eccc)
        match = source.new_match(
            station_id=SARITA,
            name="SARITA RIVER NEAR BAMFIELD",
            lat=SARITA_LAT,
            lon=SARITA_LON,
        )
        series = source.fetch_station(Query.from_area(BARKLEY_SOUND, WEEK_2024), match)

        assert series is not None and not series.is_empty
        assert len(series.frame) == 8
        assert series.frame.index[0] == pd.Timestamp("2024-07-01T00:00:00Z")
        # Sarita in July: a metre and a half of stage, a couple of cumecs of summer low flow.
        level = series.frame[LEVEL_CF]
        discharge = series.frame[DISCHARGE_CF]
        assert 1.0 < level.min() and level.max() < 3.0
        assert 0.0 < discharge.min() and discharge.max() < 20.0
        assert series.var_attrs[LEVEL_CF]["cell_methods"] == "time: mean"

    def test_the_realtime_collection_has_nothing_for_2024(self, eccc):
        """The bug these sources close: the same window against realtime is simply empty."""
        source = EcccHydrometric(eccc)
        match = source.new_match(
            station_id=SARITA,
            name="SARITA RIVER NEAR BAMFIELD",
            lat=SARITA_LAT,
            lon=SARITA_LON,
        )
        series = source.fetch_station(Query.from_area(BARKLEY_SOUND, WEEK_2024), match)
        assert series is None or series.is_empty

    def test_annual_peaks_returns_a_flood_bigger_than_the_daily_mean(self, eccc):
        peaks = EcccHydrometricAnnualPeaks(eccc)
        stats = EcccHydrometricAnnualStatistics(eccc)
        query = Query.from_area(BARKLEY_SOUND, YEARS_2018_2020)
        at = pd.Timestamp("2018-01-01T00:00:00Z")

        def frame(source):
            match = source.new_match(
                station_id=SARITA, name="SARITA", lat=SARITA_LAT, lon=SARITA_LON
            )
            return source.fetch_station(query, match).frame

        peak_frame, stat_frame = frame(peaks), frame(stats)
        assert len(peak_frame) == 3 and len(stat_frame) == 3
        assert peak_frame.loc[at, f"{DISCHARGE_CF}_max"] > stat_frame.loc[at, f"{DISCHARGE_CF}_max"]
        assert peak_frame.loc[at, f"{LEVEL_CF}_max_time"].endswith("Z")

    def test_discovery_finds_the_discontinued_nahmint_gauge(self, eccc):
        source = EcccHydrometricDailyMean(eccc)
        query = Query.from_area(BARKLEY_SOUND, ("1980-01-01", "1980-01-08"))
        found = {match.station_id for match in source.discover(query)}
        assert NAHMINT in found, "a discontinued gauge must still be discoverable for HYDAT"

    def test_a_mid_year_window_still_finds_the_annual_row(self, eccc):
        """Upstream returns nothing for this window unless it is grown to the whole year."""
        source = EcccHydrometricAnnualStatistics(eccc)
        match = source.new_match(
            station_id=SARITA, name="SARITA", lat=SARITA_LAT, lon=SARITA_LON
        )
        query = Query.from_area(BARKLEY_SOUND, ("2020-06-01", "2020-09-30"))
        series = source.fetch_station(query, match)
        assert list(series.frame.index) == [pd.Timestamp("2020-01-01T00:00:00Z")]
