"""ECCC hydrometric: water level and river discharge, realtime and historical.

Five collections, one gauge network. ``hydrometric-realtime`` holds roughly the last 30 days; the
four HYDAT collections hold the archive behind it. Without them a query for 2024 river data comes
back empty, which looks exactly like "no gauge here".

The historical four publish *aggregates*, not instants, and that changes three things:

1. A row covers a whole day, month or year, so the query window is grown out to whole periods
   before it is sent. Confirmed live: ``hydrometric-monthly-mean`` returns nothing at all for
   ``2020-07-15/2020-07-20`` and one row for ``2020-07-01/2020-07-31``.
2. ``cell_methods`` is set on every variable, so that an annual maximum and a daily mean are not
   resampled the same way.
3. The two annual collections publish one row per quantity per year, and those rows collide on
   time — so they are widened to one row a year before anything else touches them. See
   :func:`_pivot_by_year`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from ... import cf
from ...query import Query
from ..base import StationMatch, StationSeries
from ..ogc import OgcFeaturesSource

__all__ = [
    "EcccHydrometric",
    "EcccHydrometricDailyMean",
    "EcccHydrometricMonthlyMean",
    "EcccHydrometricAnnualStatistics",
    "EcccHydrometricAnnualPeaks",
]


class EcccHydrometric(OgcFeaturesSource):
    """Realtime water level and discharge from the national hydrometric network."""

    name = "eccc_hydrometric"
    title = "ECCC hydrometric realtime"
    node_path = "in_situ/hydrometric"
    collection = "hydrometric-realtime"
    station_collection = "hydrometric-stations"
    station_id_field = "STATION_NUMBER"
    catalogue_id_field = "STATION_NUMBER"
    time_field = "DATETIME"
    skip_fields = frozenset(
        {
            "IDENTIFIER",
            "STATION_NUMBER",
            "STATION_NAME",
            "PROV_TERR_STATE_LOC",
            "DATETIME",
            "DATETIME_LST",
        }
    )
    qc_suffix = ""
    #: A rolling window. Older records are served by the hydrometric-daily-mean /
    #: -monthly-mean / -annual-* sources below, so a historical query gets pointed there
    #: instead of an empty tree that reads as "there is no gauge here".
    retention = pd.Timedelta(days=30)
    samples_per_day = 24.0 * 4  # 15-minute reporting is typical

    fields = {
        "LEVEL": cf.FieldSpec(
            var="water_surface_height_above_reference_datum",
            standard_name="water_surface_height_above_reference_datum",
            units="m", long_name="Water level", qc_field="LEVEL_SYMBOL_EN",
        ),
        "DISCHARGE": cf.FieldSpec(
            var="water_volume_transport_in_river_channel",
            standard_name="water_volume_transport_in_river_channel",
            units="m3 s-1", long_name="River discharge", qc_field="DISCHARGE_SYMBOL_EN",
        ),
    }

    def is_qc_field(self, raw: str) -> bool:
        return raw.endswith(("_SYMBOL_EN", "_SYMBOL_FR"))

    def qc_field_for(self, raw: str, spec: cf.FieldSpec) -> str | None:
        return spec.qc_field

    def station_from_feature(
        self, query: Query, feature: Mapping[str, Any]
    ) -> StationMatch | None:
        match = super().station_from_feature(query, feature)
        if match is None:
            return None
        props = feature.get("properties") or {}
        # The catalogue lists discontinued gauges too; realtime data only exists for active ones.
        if props.get("REAL_TIME") in (0, "0", False):
            return None
        match.extra["vertical_datum"] = props.get("VERTICAL_DATUM") or None
        match.extra["status"] = props.get("STATUS_EN")
        return match

    def node_attrs(self, query: Query, match: StationMatch) -> dict[str, Any]:
        attrs = super().node_attrs(query, match)
        datum = match.extra.get("vertical_datum")
        if datum:
            attrs["datum"] = datum
        return attrs


# --------------------------------------------------------------------------- historical (HYDAT)

#: The CF names the whole network speaks. Both are in the CF standard name table; there is no
#: standard name for a peak *time*, which is why those variables carry only a long_name.
LEVEL_CF = "water_surface_height_above_reference_datum"
DISCHARGE_CF = "water_volume_transport_in_river_channel"

#: Columns that name or place a station rather than measure anything.
_IDENTITY = frozenset({"IDENTIFIER", "STATION_NAME", "STATION_NUMBER", "PROV_TERR_STATE_LOC"})


class _HydrometricHistorical(OgcFeaturesSource):
    """Shared shape for the four HYDAT collections.

    All four key on ``STATION_NUMBER`` against the same station catalogue, flag values with
    bilingual ``*_SYMBOL_EN``/``*_SYMBOL_FR`` siblings, and label each row with the period it
    covers rather than an instant.

    Two deliberate differences from :class:`EcccHydrometric`. There is no ``REAL_TIME`` filter —
    HYDAT is largely *discontinued* gauges, and five of the seven stations around Barkley Sound
    have ``REAL_TIME`` 0 while still holding decades of record. And ``record_period`` keeps the
    base default of "unknown", because ``hydrometric-stations`` publishes no first/last dates for
    any of its datasets; every station in the box is a candidate and the fetch decides.
    """

    station_collection = "hydrometric-stations"
    station_id_field = "STATION_NUMBER"
    catalogue_id_field = "STATION_NUMBER"
    skip_fields = _IDENTITY
    qc_suffix = ""

    #: pandas period alias for the interval one row covers: ``D``, ``M`` or ``Y``.
    period = "D"
    #: How this source's timestamps should be read; recorded on every node it produces.
    time_reference = ""

    # ------------------------------------------------------------------ period windows

    def period_count(self, query: Query) -> int:
        """How many periods the query touches — the exact row count for one station."""
        start, end = self.period_window(query)
        return len(pd.period_range(start=start, end=end, freq=self.period))

    def datetime_param(self, query: Query) -> str:
        start, end = self.period_window(query)
        return f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

    # ------------------------------------------------------------------ shaping

    def reshape_rows(self, rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        """Rows as the generic shaping code should see them. Identity unless overridden."""
        return rows

    def series_from_rows(
        self,
        query: Query,
        match: StationMatch,
        rows: list[Mapping[str, Any]],
        features: list[Mapping[str, Any]] | None = None,
    ) -> StationSeries | None:
        # The base trims to query.start/query.end, so it is handed the widened window instead;
        # otherwise a period overlapping the request is dropped. See period_window().
        start, end = self.period_window(query)
        return super().series_from_rows(
            query.replace(start=start, end=end), match, self.reshape_rows(rows), features
        )

    def extract_time(self, row: Mapping[str, Any]) -> Any:
        """Label each aggregate with the first instant of the period it covers.

        The time field holds a period rather than a timestamp — ``2024-07-01`` for a day,
        ``2020-01`` for a month, ``2018`` for a year — so it is resolved through the period
        instead of being parsed as an instant.
        """
        value = row.get(self.time_field)
        if not value:
            return None
        try:
            start = pd.Period(str(value)[:10], freq=self.period).start_time
        except (ValueError, TypeError):
            return None
        return start.strftime("%Y-%m-%dT%H:%M:%SZ")

    def is_qc_field(self, raw: str) -> bool:
        return raw.endswith(("_SYMBOL_EN", "_SYMBOL_FR"))

    def qc_field_for(self, raw: str, spec: cf.FieldSpec) -> str | None:
        return spec.qc_field

    # ------------------------------------------------------------------ stations

    def station_from_feature(
        self, query: Query, feature: Mapping[str, Any]
    ) -> StationMatch | None:
        match = super().station_from_feature(query, feature)
        if match is None:
            return None
        props = feature.get("properties") or {}
        match.extra["vertical_datum"] = props.get("VERTICAL_DATUM") or None
        match.extra["status"] = props.get("STATUS_EN")
        # samples_per_day still describes the cadence, but the Catalog's estimate is days x
        # cadence, which floors to zero as soon as a period is longer than the window: one year
        # of annual peaks comes out at int(365 / 365.25) = 0. A catalogue reading "~0 rows" is
        # the same "nothing here" these sources exist to stop showing, and the period count is
        # both exact and free.
        match.n_rows_est = self.period_count(query)
        return match

    def node_attrs(self, query: Query, match: StationMatch) -> dict[str, Any]:
        attrs = super().node_attrs(query, match)
        datum = match.extra.get("vertical_datum")
        if datum:
            attrs["datum"] = datum
        status = match.extra.get("status")
        if status:
            attrs["station_status"] = status
        if self.time_reference:
            attrs["time_reference"] = self.time_reference
        return attrs


class EcccHydrometricDailyMean(_HydrometricHistorical):
    """Daily mean water level and discharge — the workhorse of the historical archive."""

    name = "eccc_hydrometric_daily"
    title = "ECCC hydrometric daily mean (HYDAT)"
    node_path = "in_situ/hydrometric_daily"
    collection = "hydrometric-daily-mean"
    #: Not ``DATETIME`` as in the realtime collection — this one publishes a calendar ``DATE``.
    time_field = "DATE"
    period = "D"
    samples_per_day = 1.0
    time_reference = (
        "DATE: the mean over one local calendar day, stamped at that day's 00:00Z."
    )

    fields = {
        "LEVEL": cf.FieldSpec(
            var=LEVEL_CF, standard_name=LEVEL_CF, units="m", cell_methods="time: mean",
            long_name="Daily mean water level", qc_field="LEVEL_SYMBOL_EN",
        ),
        "DISCHARGE": cf.FieldSpec(
            var=DISCHARGE_CF, standard_name=DISCHARGE_CF, units="m3 s-1",
            cell_methods="time: mean", long_name="Daily mean river discharge",
            qc_field="DISCHARGE_SYMBOL_EN",
        ),
    }


class EcccHydrometricMonthlyMean(_HydrometricHistorical):
    """Monthly mean water level and discharge."""

    name = "eccc_hydrometric_monthly"
    title = "ECCC hydrometric monthly mean (HYDAT)"
    node_path = "in_situ/hydrometric_monthly"
    collection = "hydrometric-monthly-mean"
    #: ``DATE`` here is a year-month with no day at all: ``"2020-01"``.
    time_field = "DATE"
    period = "M"
    samples_per_day = 1.0 / 30.44
    time_reference = "DATE: a year-month, stamped at 00:00Z on the first of that month."

    fields = {
        "MONTHLY_MEAN_LEVEL": cf.FieldSpec(
            var=LEVEL_CF, standard_name=LEVEL_CF, units="m", cell_methods="time: mean",
            long_name="Monthly mean water level",
        ),
        "MONTHLY_MEAN_DISCHARGE": cf.FieldSpec(
            var=DISCHARGE_CF, standard_name=DISCHARGE_CF, units="m3 s-1",
            cell_methods="time: mean", long_name="Monthly mean river discharge",
        ),
    }


# --------------------------------------------------------------------------- annual reshaping

#: ``DATA_TYPE_EN`` is the only thing in an annual row that says whether its numbers are metres
#: or cumecs. Both collections use exactly these two values.
_DATA_TYPE_PREFIX = {"Water Level": "LEVEL", "Discharge": "DISCHARGE"}

_PEAK_CODE = {"Maximum": "MAX", "Minimum": "MIN"}

_QUANTITIES = (
    ("LEVEL", LEVEL_CF, "m", "water level"),
    ("DISCHARGE", DISCHARGE_CF, "m3 s-1", "river discharge"),
)

_EXTREMES = (("MAX", "maximum", "time: maximum"), ("MIN", "minimum", "time: minimum"))

#: No annual row carries a year field, so it is dug out of the IDENTIFIER, which looks like
#: ``08HB014.2011.level-niveaux``.
_YEAR_IN_IDENTIFIER = re.compile(r"\.(\d{4})\.")


def _extreme_fields(measured: str, when: str) -> dict[str, cf.FieldSpec]:
    """The eight columns :func:`_pivot_by_year` produces: a value and its date, per extreme.

    Built rather than written out longhand because the two annual collections differ only in
    wording — the CF names, units and cell_methods are identical — and eight hand-copied
    FieldSpecs is eight chances to paste the wrong standard name onto the wrong quantity.

    The ``cell_methods`` here are what stop :func:`omnisea.align` interpolating an annual
    extreme as though it were a sample.
    """
    table: dict[str, cf.FieldSpec] = {}
    for prefix, standard_name, units, label in _QUANTITIES:
        for code, word, cell_methods in _EXTREMES:
            table[f"{prefix}_{code}"] = cf.FieldSpec(
                var=f"{standard_name}_{code.lower()}",
                standard_name=standard_name,
                units=units,
                cell_methods=cell_methods,
                long_name=f"Annual {word} {measured} {label}",
                qc_field=f"{prefix}_{code}_SYMBOL_EN",
            )
            table[f"{prefix}_{code}_DATE"] = cf.FieldSpec(
                var=f"{standard_name}_{code.lower()}_time",
                standard_name="",  # a timestamp is not a measured quantity
                long_name=f"{when} of the annual {word} {label}",
            )
    return table


def _pivot_by_year(
    rows: list[Mapping[str, Any]],
    fill: Callable[[Mapping[str, Any], str, dict[str, Any]], None],
) -> list[Mapping[str, Any]]:
    """Collapse HYDAT's long annual rows into one row per year.

    Both annual collections repeat a station's year once per quantity, and the repeats carry the
    *same* timestamp: at Sarita River the 2018 water-level maximum and the 2018 discharge maximum
    are both stamped ``2018-01-21T09:00``, being the same flood measured two ways. Left long,
    ``frame_from_records`` would keep only the last row at that instant and the level peak would
    vanish without a trace. Widening each year into one row with a column per quantity keeps
    both.
    """
    wide: dict[str, dict[str, Any]] = {}
    for props in rows:
        year = _record_year(props)
        prefix = _DATA_TYPE_PREFIX.get(str(props.get("DATA_TYPE_EN")))
        if year is None or prefix is None:
            continue
        fill(props, prefix, wide.setdefault(year, {"YEAR": year}))
    return [wide[year] for year in sorted(wide)]


def _fill_annual_statistics(props: Mapping[str, Any], prefix: str, out: dict[str, Any]) -> None:
    """One ``hydrometric-annual-statistics`` row: both extremes of the daily values."""
    for code in ("MAX", "MIN"):
        out[f"{prefix}_{code}"] = props.get(f"{code}_VALUE")
        out[f"{prefix}_{code}_DATE"] = props.get(f"{code}_DATE") or None
        # This collection writes "no flag" as an empty string where the others write null (1929
        # of 2000 sampled rows), which would otherwise survive as a column of empty strings.
        out[f"{prefix}_{code}_SYMBOL_EN"] = props.get(f"{code}_SYMBOL_EN") or None


def _fill_annual_peaks(props: Mapping[str, Any], prefix: str, out: dict[str, Any]) -> None:
    """One ``hydrometric-annual-peaks`` row: a single extreme, named by ``PEAK_CODE_EN``."""
    code = _PEAK_CODE.get(str(props.get("PEAK_CODE_EN")))
    if code is None:
        return
    out[f"{prefix}_{code}"] = props.get("PEAK")
    out[f"{prefix}_{code}_DATE"] = _peak_instant_utc(props)
    out[f"{prefix}_{code}_SYMBOL_EN"] = props.get("SYMBOL_EN") or None


def _record_year(props: Mapping[str, Any]) -> str | None:
    """The calendar year a HYDAT annual record belongs to.

    Neither annual collection publishes a year field. ``hydrometric-annual-statistics`` has no
    single date at all — only MIN_DATE and MAX_DATE, and 7 rows in 600 sampled have both null —
    while one ``hydrometric-annual-peaks`` row in roughly two thousand has a null DATE. The year
    is always present in the IDENTIFIER, so that is read first and the record's own dates are
    the fallback. Station names containing a period (``ST. FRANCIS RIVER AT ...``) are why this
    searches for a four-digit component rather than splitting on dots.
    """
    found = _YEAR_IN_IDENTIFIER.search(str(props.get("IDENTIFIER") or ""))
    if found:
        return found.group(1)
    for key in ("DATE", "MAX_DATE", "MIN_DATE"):
        year = str(props.get(key) or "")[:4]
        if year.isdigit():
            return year
    return None


def _peak_instant_utc(props: Mapping[str, Any]) -> str | None:
    """When a peak happened, in UTC.

    ``hydrometric-annual-peaks`` stamps DATE in local standard time and states the offset in a
    separate field: ``"2018-01-21T09:00"`` with ``TIMEZONE_OFFSET`` ``"-8"``. Reading that as
    UTC would move every peak on the west coast by eight hours. DATE is occasionally only a
    date, or a bare year, and once in about two thousand rows it is null.
    """
    raw = props.get("DATE")
    if not raw:
        return None
    try:
        instant = pd.Timestamp(str(raw))
        offset = props.get("TIMEZONE_OFFSET")
        if offset not in (None, ""):
            instant -= pd.Timedelta(hours=float(offset))
    except (ValueError, TypeError):
        return None
    return instant.strftime("%Y-%m-%dT%H:%M:%SZ")


class EcccHydrometricAnnualStatistics(_HydrometricHistorical):
    """Annual maximum and minimum of the *daily* values, with the date each extreme fell on."""

    name = "eccc_hydrometric_annual"
    title = "ECCC hydrometric annual statistics (HYDAT)"
    node_path = "in_situ/hydrometric_annual"
    collection = "hydrometric-annual-statistics"
    #: Synthesised by the pivot. This collection has no time field of its own — only MAX_DATE
    #: and MIN_DATE, either of which can be null.
    time_field = "YEAR"
    period = "Y"
    samples_per_day = 1.0 / 365.25
    time_reference = (
        "One row per calendar year, stamped at 00:00Z on 1 January of that year. The dates the "
        "extremes actually fell on are carried as the *_max_time and *_min_time variables."
    )

    fields = _extreme_fields("daily mean", "Date")

    def reshape_rows(self, rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return _pivot_by_year(rows, _fill_annual_statistics)


class EcccHydrometricAnnualPeaks(_HydrometricHistorical):
    """Annual maximum and minimum *instantaneous* level and discharge, with the peak times.

    The instantaneous peak is what a flood-frequency analysis wants. The annual statistic in
    :class:`EcccHydrometricAnnualStatistics` is a mean over the day the peak fell in, so it is
    always the smaller number of the two.
    """

    name = "eccc_hydrometric_annual_peaks"
    title = "ECCC hydrometric annual peaks (HYDAT)"
    node_path = "in_situ/hydrometric_annual_peaks"
    collection = "hydrometric-annual-peaks"
    time_field = "YEAR"  # synthesised by the pivot, as for the annual statistics
    period = "Y"
    samples_per_day = 1.0 / 365.25
    time_reference = (
        "One row per calendar year, stamped at 00:00Z on 1 January of that year. The instant of "
        "each peak is carried as a *_max_time or *_min_time variable, converted to UTC from the "
        "local standard time the collection publishes."
    )

    fields = _extreme_fields("instantaneous", "UTC time")

    def reshape_rows(self, rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return _pivot_by_year(rows, _fill_annual_peaks)
