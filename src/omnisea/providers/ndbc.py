"""NOAA NDBC — moored buoys: waves, marine wind, and sea surface conditions.

``https://www.ndbc.noaa.gov``

The National Data Buoy Center is where wave data lives. No other source in the registry
serves significant wave height, wave period or wave direction natively, and the buoys also
carry marine wind, pressure and sea surface temperature offshore where no tide gauge or
climate station stands. The station table includes buoys NDBC relays for partner programs —
dozens of Environment and Climate Change Canada buoys among them — so the same source answers
on both sides of the border.

Three file families cover any window, all plain text, none needing a key:

- ``data/realtime2/{id}.txt`` — the last 45 days at native cadence;
- ``data/stdmet/{Mon}/{id}{m}{year}.txt.gz`` — month files for the current year;
- ``data/historical/stdmet/{id}h{year}.txt.gz`` — one file per archived year.

A file that does not exist answers 404, which means "this buoy has no records there" —
partner-relay buoys, for instance, appear in realtime but archive with their home agency.
Every value is UTC and already SI. Gaps are all-nines sentinels (``MM``, 99.0, 999, 9999)
at the column's own precision.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import pandas as pd

from .. import cf
from ..errors import UpstreamError
from ..http import DEFAULT_MAX_WORKERS, NEVER_CACHE, get_text, map_threads
from ..query import Query
from .base import Provider, RetrievalSource, StationMatch, StationSeries

log = logging.getLogger("omnisea.ndbc")

__all__ = ["NdbcProvider"]

BASE = "https://www.ndbc.noaa.gov"
STATION_TABLE = f"{BASE}/data/stations/station_table.txt"
FILE_READER = f"{BASE}/view_text_file.php"

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

#: ``48.334 N 123.579 W`` (an optional parenthesised DMS repeat follows; ignored).
_LOCATION = re.compile(r"([0-9.]+)\s*([NS])\s+([0-9.]+)\s*([EW])")

#: NDBC marks a gap with all nines at the column's own precision.
_SENTINELS = {99.0, 999.0, 9999.0}


class NdbcProvider(Provider):
    name = "ndbc"
    title = "NOAA National Data Buoy Center"
    base_url = BASE
    license = "US Government work — public domain (NOAA/NDBC)"
    terms_url = "https://www.weather.gov/disclaimer"

    cache_policy = {
        "www.ndbc.noaa.gov/data/stations/*": timedelta(days=7),
        # Archived years never change; current-year month files grow daily.
        "www.ndbc.noaa.gov/view_text_file.php*historical*": timedelta(days=30),
        "www.ndbc.noaa.gov/view_text_file.php*": timedelta(hours=6),
        "www.ndbc.noaa.gov/data/realtime2/*": NEVER_CACHE,
    }

    def build_sources(self) -> Sequence[RetrievalSource]:
        return [NdbcStdmetSource(self)]

    def all_stations(self) -> list[dict[str, Any]]:
        """The station table: id, owner, name and position for ~1,900 platforms."""
        text = get_text(STATION_TABLE, None, provider=self.name)
        stations: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 7:
                continue
            place = _LOCATION.search(parts[6])
            if place is None:
                continue
            lat = float(place.group(1)) * (1 if place.group(2) == "N" else -1)
            lon = float(place.group(3)) * (1 if place.group(4) == "E" else -1)
            stations.append({
                "id": parts[0].strip(),
                "owner": parts[1].strip(),
                "kind": parts[2].strip(),
                "name": parts[4].strip(),
                "lat": lat,
                "lon": lon,
            })
        return stations


class NdbcStdmetSource(RetrievalSource):
    """Standard meteorological records: waves, wind, pressure, temperatures."""

    name = "ndbc_stdmet"
    title = "NDBC standard meteorological data"
    node_path = "in_situ/buoys"
    feature_type = "timeSeries"
    #: Ten-minute reporting is typical for moored buoys.
    samples_per_day = 144.0

    #: Keyed by the stdmet column header. Everything NDBC publishes is already SI.
    fields = {
        "WVHT": cf.FieldSpec(
            var="sea_surface_wave_significant_height",
            standard_name="sea_surface_wave_significant_height",
            units="m",
            long_name="Significant wave height",
        ),
        "DPD": cf.FieldSpec(
            var="sea_surface_wave_period_at_variance_spectral_density_maximum",
            standard_name="sea_surface_wave_period_at_variance_spectral_density_maximum",
            units="s",
            long_name="Dominant wave period",
        ),
        "APD": cf.FieldSpec(
            var="sea_surface_wave_mean_period",
            standard_name="sea_surface_wave_mean_period",
            units="s",
            long_name="Average wave period",
        ),
        "MWD": cf.FieldSpec(
            var="sea_surface_wave_from_direction",
            standard_name="sea_surface_wave_from_direction",
            units="degree",
            long_name="Wave direction at the dominant period",
        ),
        "WDIR": cf.FieldSpec(
            var="wind_from_direction",
            standard_name="wind_from_direction",
            units="degree",
            long_name="Wind direction",
        ),
        "WSPD": cf.FieldSpec(
            var="wind_speed",
            standard_name="wind_speed",
            units="m s-1",
            long_name="Wind speed",
        ),
        "GST": cf.FieldSpec(
            var="wind_speed_of_gust",
            standard_name="wind_speed_of_gust",
            units="m s-1",
            long_name="Wind gust",
        ),
        "PRES": cf.FieldSpec(
            var="air_pressure_at_mean_sea_level",
            standard_name="air_pressure_at_sea_level",
            units="hPa",
            long_name="Sea level pressure",
        ),
        "ATMP": cf.FieldSpec(
            var="air_temperature",
            standard_name="air_temperature",
            units="degC",
            long_name="Air temperature",
        ),
        "WTMP": cf.FieldSpec(
            var="sea_surface_temperature",
            standard_name="sea_surface_temperature",
            units="degC",
            long_name="Sea surface temperature",
        ),
        "DEWP": cf.FieldSpec(
            var="dew_point_temperature",
            standard_name="dew_point_temperature",
            units="degC",
            long_name="Dew point",
        ),
    }

    def locate(self, station_id: str) -> tuple[float, float, str] | None:
        wanted = station_id.lower()
        for st in self.provider.all_stations():
            if st["id"].lower() == wanted:
                return st["lat"], st["lon"], st["name"] or station_id
        return None

    # ------------------------------------------------------------------ discovery

    def discover(self, query: Query) -> list[StationMatch]:
        matches: list[StationMatch] = []
        for st in self.provider.all_stations():
            if not query.contains(st["lat"], st["lon"]):
                continue
            # The station table states no period of record; absent years answer 404 at
            # fetch time, so availability is settled where it is actually known.
            matches.append(
                self.new_match(
                    station_id=st["id"],
                    name=st["name"] or st["id"],
                    lat=st["lat"],
                    lon=st["lon"],
                    variables=tuple(sorted(spec.var for spec in self.fields.values())),
                    n_rows_est=self.row_estimate(query),
                    extra={"owner": st["owner"], "kind": st["kind"]},
                ).attach_site(query)
            )
        log.debug("ndbc_stdmet discovered %d buoy(s)", len(matches))
        return matches

    # ------------------------------------------------------------------ retrieval

    def fetch(self, query: Query, matches: list[StationMatch]) -> list[StationSeries]:
        results = map_threads(
            lambda match: self._fetch_buoy(query, match),
            matches,
            max_workers=int(query.option("max_workers", DEFAULT_MAX_WORKERS)),
            label="ndbc buoys",
        )
        return [r for r in results if r is not None]

    def _fetch_buoy(self, query: Query, match: StationMatch) -> StationSeries | None:
        now = pd.Timestamp.now(tz="UTC")
        frames: list[pd.DataFrame] = []

        # Archived years, then this year's month files, then the 45-day realtime window.
        # They overlap at the seams by design; the concat below keeps the first answer.
        sid = match.station_id.lower()
        for year in range(query.start.year, query.end.year + 1):
            if year < now.year:
                frames.append(self._read_file(
                    {"filename": f"{sid}h{year}.txt.gz", "dir": "data/historical/stdmet/"}
                ))
            else:
                for month in range(1, 13):
                    stamp = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
                    if stamp > now or stamp + pd.offsets.MonthEnd(1) < query.start:
                        continue
                    frames.append(self._read_file(
                        {"filename": f"{sid}{month}{year}.txt.gz",
                         "dir": f"data/stdmet/{MONTH_ABBR[month - 1]}/"}
                    ))
        if query.end > now - timedelta(days=45):
            frames.append(self._read_realtime(sid))

        usable = [f for f in frames if f is not None and not f.empty]
        if usable:
            frame = pd.concat(usable)
            frame = frame[~frame.index.duplicated(keep="first")].sort_index()
            frame = frame.loc[(frame.index >= query.start) & (frame.index <= query.end)]
        else:
            frame = pd.DataFrame()

        var_attrs = {
            spec.var: cf.cf_attrs(spec, to_cf_units=False)
            for spec in self.fields.values()
            if spec.var in frame.columns
        }
        return StationSeries(
            match=match,
            frame=frame,
            node_path=f"{self.node_path}/{match.station_id}",
            attrs=self.base_attrs(
                source_url=f"{BASE}/station_page.php?station={match.station_id}",
                platform_owner=match.extra.get("owner", ""),
                platform_kind=match.extra.get("kind", ""),
            ),
            var_attrs=var_attrs,
        )

    def _read_file(self, params: dict[str, str]) -> pd.DataFrame | None:
        try:
            text = get_text(FILE_READER, params, provider=self.name)
        except UpstreamError as exc:
            if exc.status == 404:
                return None  # this buoy has no records in that file — an answer
            raise
        return self._parse_stdmet(text)

    def _read_realtime(self, sid: str) -> pd.DataFrame | None:
        try:
            text = get_text(f"{BASE}/data/realtime2/{sid.upper()}.txt", None,
                            provider=self.name)
        except UpstreamError as exc:
            if exc.status == 404:
                return None
            raise
        return self._parse_stdmet(text)

    def _parse_stdmet(self, text: str) -> pd.DataFrame:
        """Whitespace table; header row, a units row on newer files, ``MM``/all-nines gaps.

        Files older than 2007 have four date columns (two-digit or four-digit year, no
        minutes); everything since has five. The header names the layout, so the parse
        follows the header rather than assuming an era.
        """
        lines = text.splitlines()
        if not lines:
            return pd.DataFrame()
        header = lines[0].lstrip("#").split()
        body = "\n".join(line for line in lines[1:] if not line.startswith("#"))
        if not body.strip():
            return pd.DataFrame()
        table = pd.read_csv(io.StringIO(body), sep=r"\s+", names=header, na_values=["MM"])

        year_col = "YYYY" if "YYYY" in header else "YY"
        years = table[year_col].astype(int)
        years = years.where(years > 100, years + 1900)
        stamp = pd.to_datetime(
            {
                "year": years,
                "month": table["MM"].astype(int),
                "day": table["DD"].astype(int),
                "hour": table["hh"].astype(int),
                "minute": table["mm"].astype(int) if "mm" in header else 0,
            },
            utc=True,
        )

        out = pd.DataFrame(index=stamp)
        out.index.name = "time"
        for column, spec in self.fields.items():
            if column not in table.columns:
                continue
            values = pd.to_numeric(table[column].values, errors="coerce")
            values = pd.Series(values, index=stamp)
            out[spec.var] = values.where(~values.isin(_SENTINELS))
        return out.dropna(how="all")
