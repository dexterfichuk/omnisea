"""Columbia Basin Research DART — dam operations and adult salmon passage.

``https://www.cbr.washington.edu/dart``

The University of Washington's Data Access in Real Time serves the Columbia and Snake River
hydrosystem: daily project outflow, spill, inflow and total dissolved gas at each mainstem
dam, and the **adult fish ladder counts** — Chinook, steelhead, sockeye, coho, shad, lamprey —
that make this basin's data unlike anywhere else's. Bonneville's count record starts in 1938;
no other source in this registry has fish.

Two shapes, both plain CSV, no key:

- the river-environment report (``mg.php``), one column per ``year:project:series``;
- the adult-passage report (``adult_daily.php``), one row per day with species columns and
  the ladder's own water temperature.

Both label rows by the project's **local calendar date** — the same convention as ECCC's and
USGS's daily archives, handled the same way. Fish counts have no CF standard name and never
will; they are served under plain English variable names with ``cell_methods: "time: sum"``,
because a day's count is a day's total, not a sample.

The dams themselves are the station catalogue: thirteen projects, positions fixed in
concrete. Requests chunk by calendar year because every DART report is year-keyed.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import pandas as pd

from .. import cf
from ..http import DEFAULT_MAX_WORKERS, get_text, map_threads
from ..query import Query
from .base import Provider, RetrievalSource, StationMatch, StationSeries

log = logging.getLogger("omnisea.cbr")

__all__ = ["CbrProvider"]

RIVER_REPORT = "https://www.cbr.washington.edu/dart/cs/php/rpt/mg.php"
ADULT_REPORT = "https://www.cbr.washington.edu/dart/cs/php/rpt/adult_daily.php"

#: The mainstem hydro projects DART reports, positions fixed in concrete. Codes are DART's
#: own; coordinates are the dam structures themselves (public engineering records).
DAMS: dict[str, tuple[str, float, float]] = {
    "BON": ("Bonneville Dam", 45.644, -121.941),
    "TDA": ("The Dalles Dam", 45.615, -121.134),
    "JDA": ("John Day Dam", 45.716, -120.693),
    "MCN": ("McNary Dam", 45.936, -119.297),
    "PRD": ("Priest Rapids Dam", 46.645, -119.910),
    "WAN": ("Wanapum Dam", 46.876, -119.973),
    "RIS": ("Rock Island Dam", 47.343, -120.094),
    "RRH": ("Rocky Reach Dam", 47.530, -120.300),
    "WEL": ("Wells Dam", 47.947, -119.863),
    "IHR": ("Ice Harbor Dam", 46.250, -118.880),
    "LMN": ("Lower Monumental Dam", 46.563, -118.540),
    "LGS": ("Little Goose Dam", 46.587, -118.026),
    "LWG": ("Lower Granite Dam", 46.660, -117.428),
}


class CbrProvider(Provider):
    name = "cbr"
    title = "Columbia Basin Research (University of Washington) — DART"
    base_url = "https://www.cbr.washington.edu/dart"
    license = (
        "Public data compiled by Columbia Basin Research from USACE, PUD and agency records; "
        "credit Columbia Basin Research, University of Washington"
    )
    terms_url = "https://www.cbr.washington.edu/dart"

    cache_policy = {
        # Year-keyed daily reports: past years are settled; the current year grows daily.
        "www.cbr.washington.edu/dart/cs/php/rpt/*": timedelta(hours=6),
    }

    def build_sources(self) -> Sequence[RetrievalSource]:
        return [DartRiverSource(self), DartPassageSource(self)]


class _DartSource(RetrievalSource):
    """Shared shape: a fixed dam catalogue, year-keyed CSV reports, local-date stamps."""

    feature_type = "timeSeries"
    period = "D"
    samples_per_day = 1.0

    def locate(self, station_id: str) -> tuple[float, float, str] | None:
        dam = DAMS.get(str(station_id).upper())
        if dam is None:
            return None
        name, lat, lon = dam
        return lat, lon, name

    def discover(self, query: Query) -> list[StationMatch]:
        matches: list[StationMatch] = []
        for code, (name, lat, lon) in DAMS.items():
            if not query.contains(lat, lon):
                continue
            matches.append(
                self.new_match(
                    station_id=code,
                    name=name,
                    lat=lat,
                    lon=lon,
                    variables=tuple(sorted(spec.var for spec in self.fields.values())),
                    n_rows_est=self.row_estimate(query),
                ).attach_site(query)
            )
        return matches

    def fetch(self, query: Query, matches: list[StationMatch]) -> list[StationSeries]:
        results = map_threads(
            lambda match: self._fetch_dam(query, match),
            matches,
            max_workers=int(query.option("max_workers", DEFAULT_MAX_WORKERS)),
            label="dart dams",
        )
        return [r for r in results if r is not None]

    def _fetch_dam(self, query: Query, match: StationMatch) -> StationSeries | None:
        frames = [
            frame
            for year in range(query.start.year, query.end.year + 1)
            if (frame := self._read_year(query, match, year)) is not None and not frame.empty
        ]
        if frames:
            frame = pd.concat(frames).sort_index()
            frame = frame[~frame.index.duplicated(keep="first")]
            # Local calendar dates, stamped naive at midnight: trimmed by *date*, so the last
            # requested day survives even when the window's UTC end lands mid-afternoon.
            frame = frame.loc[
                (frame.index >= query.start.tz_localize(None).normalize())
                & (frame.index <= query.end.tz_localize(None))
            ]
            frame.index = frame.index.tz_localize("UTC")
            frame.index.name = "time"
        else:
            frame = pd.DataFrame()

        to_cf = self.to_cf_units(query)
        if to_cf:
            # The parse writes the report's own numbers; the conversion happens here, beside
            # the attrs that claim it — so the units label and the values can never disagree.
            for spec in self.fields.values():
                if spec.var in frame.columns:
                    frame[spec.var] = [
                        cf.convert(v, spec, to_cf_units=True) for v in frame[spec.var]
                    ]
        var_attrs = {
            spec.var: cf.cf_attrs(spec, to_cf_units=to_cf)
            for spec in self.fields.values()
            if spec.var in frame.columns
        }
        for name, attrs in self.extra_var_attrs(frame).items():
            var_attrs[name] = attrs
        return StationSeries(
            match=match,
            frame=frame,
            node_path=f"{self.node_path}/{match.station_id}",
            attrs=self.base_attrs(
                source_url=self.report_url(match),
                time_reference=(
                    "LOCAL_DATE: daily rows are labelled by the project's local calendar "
                    "date and stamped at midnight with no UTC offset."
                ),
            ),
            var_attrs=var_attrs,
        )

    def extra_var_attrs(self, frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
        return {}

    def report_url(self, match: StationMatch) -> str:
        raise NotImplementedError

    def _read_year(
        self, query: Query, match: StationMatch, year: int
    ) -> pd.DataFrame | None:
        raise NotImplementedError


#: The report's own column short-names -> the request series they answer for.
_REPORT_NAMES = {
    "outflow": "Outflow",
    "inflow": "Inflow",
    "spill": "Spill",
    "disgas": "Dissolved Gas",
}


class DartRiverSource(_DartSource):
    """Daily project outflow, spill, inflow and total dissolved gas at each dam."""

    name = "dart_river"
    title = "DART river environment (daily)"
    node_path = "in_situ/hydrometric_daily"

    #: Keyed by DART's report series name. Outflow is the discharge past the project — the
    #: same quantity a USGS gauge below the dam measures, under the same variable name.
    fields = {
        "Outflow": cf.FieldSpec(
            var="river_discharge",
            standard_name="water_volume_transport_in_river_channel",
            units="kcfs",
            long_name="Project outflow (daily mean)",
            cf_units="m3 s-1",
            cf_scale=28.316846592,
            cell_methods="time: mean",
        ),
        "Inflow": cf.FieldSpec(
            var="river_discharge_inflow",
            standard_name="",
            units="kcfs",
            long_name="Project inflow (daily mean)",
            cf_units="m3 s-1",
            cf_scale=28.316846592,
            cell_methods="time: mean",
        ),
        "Spill": cf.FieldSpec(
            var="spill_discharge",
            standard_name="",
            units="kcfs",
            long_name="Spillway discharge (daily mean)",
            cf_units="m3 s-1",
            cf_scale=28.316846592,
            cell_methods="time: mean",
        ),
        "Dissolved Gas": cf.FieldSpec(
            var="total_dissolved_gas_pressure",
            standard_name="",
            units="mmHg",
            long_name="Total dissolved gas pressure (daily mean)",
            cell_methods="time: mean",
        ),
    }

    def report_url(self, match: StationMatch) -> str:
        return f"{RIVER_REPORT}?mgconfig=river&loc[]={match.station_id}"

    def _read_year(
        self, query: Query, match: StationMatch, year: int
    ) -> pd.DataFrame | None:
        text = get_text(
            RIVER_REPORT,
            {
                "sc": "1",
                "mgconfig": "river",
                "outputFormat": "csv",
                "year[]": str(year),
                "loc[]": match.station_id,
                "data[]": list(self.fields),
                "startdate": "1/1",
                "enddate": "12/31",
            },
            provider=self.name,
        )
        if text.lstrip().startswith("<!DOCTYPE"):
            # DART answers an unusable request with its HTML page rather than an error code.
            log.debug("dart_river %s %s: HTML answer, treated as empty", match.station_id, year)
            return None
        return self._parse(text, year)

    def _parse(self, text: str, year: int) -> pd.DataFrame | None:
        lines = text.splitlines()
        header_at = next(
            (i for i, line in enumerate(lines) if line.startswith("mm/dd")), None
        )
        if header_at is None:
            return None
        body: list[str] = []
        for line in lines[header_at:]:
            if not line.strip() or line.startswith("Notes:"):
                break
            body.append(line)
        table = pd.read_csv(io.StringIO("\n".join(body)))
        out = pd.DataFrame(
            index=pd.to_datetime(
                table["mm/dd"].astype(str) + f"/{year}", format="%m/%d/%Y"
            )
        )
        for column in table.columns:
            # Columns arrive as "2024:BON:outflow (kcfs)"; the series name is the third field.
            parts = str(column).split(":")
            if len(parts) != 3:
                continue
            series_name = parts[2].split(" (")[0].strip().lower()
            spec = self.fields.get(_REPORT_NAMES.get(series_name, ""))
            if spec is None:
                continue
            out[spec.var] = pd.to_numeric(table[column].values, errors="coerce")
        return out.dropna(how="all")


#: Adult-passage CSV columns -> variable names. DART's abbreviations, spelled out. Counts
#: are day totals at the fish ladder; jacks are early-returning juveniles counted apart.
PASSAGE_COLUMNS = {
    "Chin": ("chinook", "Adult Chinook salmon counted at the ladder"),
    "JChin": ("chinook_jack", "Jack Chinook salmon counted at the ladder"),
    "Stlhd": ("steelhead", "Steelhead counted at the ladder"),
    "WStlhd": ("steelhead_wild", "Wild (unclipped) steelhead counted at the ladder"),
    "Sock": ("sockeye", "Sockeye salmon counted at the ladder"),
    "Coho": ("coho", "Adult coho salmon counted at the ladder"),
    "JCoho": ("coho_jack", "Jack coho salmon counted at the ladder"),
    "Chum": ("chum", "Chum salmon counted at the ladder"),
    "Pink": ("pink", "Pink salmon counted at the ladder"),
    "Shad": ("shad", "American shad counted at the ladder"),
    "LmpryCombined": ("lamprey", "Pacific lamprey counted at the ladder (day + night)"),
    "BTrout": ("bull_trout", "Bull trout counted at the ladder"),
}


class DartPassageSource(_DartSource):
    """Adult fish counted at each dam's ladder, daily — Bonneville's record starts in 1938."""

    name = "dart_passage"
    title = "DART adult passage daily counts"
    node_path = "in_situ/fish_passage"

    fields = {
        column: cf.FieldSpec(
            var=var,
            standard_name="",  # there is no CF standard name for a salmon, and that is fine
            units="1",
            long_name=long_name,
            cell_methods="time: sum",
        )
        for column, (var, long_name) in PASSAGE_COLUMNS.items()
    } | {
        "TempC": cf.FieldSpec(
            var="water_temperature",
            standard_name="sea_water_temperature",
            units="degC",
            long_name="Water temperature at the ladder",
            cf_units="K",
            cf_offset=273.15,
        ),
    }

    def report_url(self, match: StationMatch) -> str:
        return f"{ADULT_REPORT}?proj={match.station_id}"

    def _read_year(
        self, query: Query, match: StationMatch, year: int
    ) -> pd.DataFrame | None:
        text = get_text(
            ADULT_REPORT,
            {
                "sc": "1",
                "outputFormat": "csv",
                "year": str(year),
                "proj": match.station_id,
                "span": "no",
                "startdate": "1/1",
                "enddate": "12/31",
                "run": "",
            },
            provider=self.name,
        )
        if text.lstrip().startswith("<!DOCTYPE"):
            log.debug(
                "dart_passage %s %s: HTML answer, treated as empty", match.station_id, year
            )
            return None
        lines = []
        for line in text.splitlines():
            if line.startswith("Notes:"):
                break
            if line.strip():
                lines.append(line)
        if len(lines) < 2:
            return None
        table = pd.read_csv(io.StringIO("\n".join(lines)))
        if "Date" not in table.columns:
            return None
        out = pd.DataFrame(index=pd.to_datetime(table["Date"], format="%Y-%m-%d"))
        for column, spec in self.fields.items():
            if column in table.columns:
                out[spec.var] = pd.to_numeric(table[column].values, errors="coerce")
        if "Chinook Run" in table.columns:
            # Sp/Su/Fa: which run the day's Chinook count belongs to — a label, not a number.
            out["chinook_run"] = table["Chinook Run"].astype(str).values
        return out.dropna(how="all")

    def extra_var_attrs(self, frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
        if "chinook_run" not in frame.columns:
            return {}
        return {
            "chinook_run": {
                "long_name": "Chinook run designation",
                "comment": "Sp spring, Su summer, Fa fall, as assigned by DART.",
                cf.MAPPED_ATTR: 0,
                "source_field": "Chinook Run",
            }
        }
