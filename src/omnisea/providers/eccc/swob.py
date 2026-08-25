"""ECCC Surface Weather Observations (SWOB), land and marine.

Two collections, one format. Both publish a ``-uom`` beside every measurement, so neither source
hardcodes a unit table; both mark quality with a ``-qa`` sibling. What differs is where the
observations live:

* Land SWOB is an OGC API - Features collection (``swob-realtime``) on ``api.weather.gc.ca``.
* **Marine SWOB is not on that API at all.** ``swob-marine-stations`` lists the 42 moored buoys,
  but ``swob-realtime`` holds none of their observations. Filtering it by a buoy's ``msc_id``,
  ``wmo_synop_id``, station name, or a bbox around its position returns zero matches over all
  time; its 674 queryables contain no wave or sea-surface field; and every record sampled from
  it carries an ``-atmospheric-surface_weather-`` dataset id. The buoy observations are
  published only as SWOB-ML XML on the MSC Datamart, one file per station per report.

:class:`EcccSwobMarine` therefore discovers over the OGC API like everything else here and then
retrieves over a different transport entirely, which is why it is the one source that replaces
``fetch`` rather than inheriting it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any
from xml.etree import ElementTree

import pandas as pd
import requests

from ... import cf
from ...errors import PayloadTooLargeError, UpstreamError
from ...http import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_TIMEOUT,
    get_session,
    map_threads,
    paginate_ogc_items,
)
from ...query import Query
from ..base import StationMatch, StationSeries
from ..ogc import OgcFeaturesSource, point_from_feature
from .common import CM_TO_M, DEGC_TO_K, HPA_TO_PA, KMH_TO_MS

log = logging.getLogger("omnisea.eccc.swob")

__all__ = ["EcccSwobRealtime", "EcccSwobMarine"]


_SWOB_SUFFIXES = ("-uom", "-qa", "-value", "-data_flag-code_src")

_SWOB_SKIP = frozenset(
    {
        "dataset",
        "id",
        "url",
        "obs_date_tm",
        "processed_date_tm",
        "lat",
        "long",
        "clim_id",
        "msc_id",
        "stn_nam",
        "tc_id",
        "wmo_synop_id",
        "data_pvdr",
        "stn_elev",
        "date_tm",
    }
)


class _SwobSource(OgcFeaturesSource):
    """What the land and marine SWOB collections share: their unit and flag conventions."""
    #: Both SWOB collections are rolling archives of roughly the last month. Declared here so a
    #: historical query is told that, rather than matching nothing and reading as "no station".
    retention = pd.Timedelta(days=30)


    qc_suffix = "-qa"
    skip_fields = _SWOB_SKIP

    def is_qc_field(self, raw: str) -> bool:
        return raw.endswith(_SWOB_SUFFIXES)

    def qc_field_for(self, raw: str, spec: cf.FieldSpec) -> str | None:
        return f"{raw}-qa"

    def units_for(self, raw: str, rows: list[Mapping[str, Any]]) -> str | None:
        """Units come from the field's own ``-uom`` sibling, not from a table."""
        key = f"{raw}-uom"
        for row in rows:
            value = row.get(key)
            if value:
                return str(value)
        return None


class EcccSwobRealtime(_SwobSource):
    """Surface Weather Observations, roughly the last 30 days at minute-to-hourly cadence.

    Units are **read from the data**: every measurement property has a ``-uom`` sibling, so this
    source never hardcodes a unit table and never has to be updated when a station changes
    sensors.

    Discovery samples a narrow time slice across the query area rather than reading a catalogue,
    even though ``swob-stations`` and ``swob-partner-stations`` exist. Measured against an hour
    of live records: the two catalogues list 2,968 stations between them but cover only 1,018 of
    the 1,040 that actually reported, and neither publishes a period of record to filter the
    silent ones out. Cataloguing would therefore miss ~2% of live stations *and* spend a station
    request on roughly two dead ones for every live one, where sampling costs a single paged
    request and enumerates exactly what is reporting.
    """

    name = "eccc_swob"
    title = "ECCC surface weather observations (SWOB) realtime"
    node_path = "in_situ/weather_realtime"
    collection = "swob-realtime"
    station_collection = ""  # catalogued, but sampling is both cheaper and more complete
    station_id_field = "msc_id-value"
    time_field = "obs_date_tm"
    samples_per_day = 24.0 * 6  # 10-minute reporting is typical

    fields = {
        "air_temp": cf.FieldSpec(
            var="air_temperature", standard_name="air_temperature",
            long_name="Air temperature", **DEGC_TO_K,
        ),
        "dwpt_temp": cf.FieldSpec(
            var="dew_point_temperature", standard_name="dew_point_temperature",
            long_name="Dew point temperature", **DEGC_TO_K,
        ),
        "wetblb_temp": cf.FieldSpec(
            var="wet_bulb_temperature", standard_name="wet_bulb_temperature",
            long_name="Wet bulb temperature", **DEGC_TO_K,
        ),
        "rel_hum": cf.FieldSpec(
            var="relative_humidity", standard_name="relative_humidity",
            long_name="Relative humidity",
        ),
        "stn_pres": cf.FieldSpec(
            var="air_pressure", standard_name="air_pressure",
            long_name="Station air pressure", **HPA_TO_PA,
        ),
        "mslp": cf.FieldSpec(
            var="air_pressure_at_mean_sea_level", standard_name="air_pressure_at_mean_sea_level",
            long_name="Mean sea level pressure", **HPA_TO_PA,
        ),
        "avg_wnd_spd_10m_pst10mts": cf.FieldSpec(
            var="wind_speed", standard_name="wind_speed",
            cell_methods="time: mean (interval: 10 minutes)",
            long_name="10 m wind speed, 10-minute mean", **KMH_TO_MS,
        ),
        "avg_wnd_dir_10m_pst10mts": cf.FieldSpec(
            var="wind_from_direction", standard_name="wind_from_direction",
            cell_methods="time: mean (interval: 10 minutes)",
            long_name="10 m wind direction, 10-minute mean",
        ),
        "max_wnd_spd_10m_pst10mts": cf.FieldSpec(
            var="wind_speed_of_gust", standard_name="wind_speed_of_gust",
            cell_methods="time: maximum (interval: 10 minutes)",
            long_name="10 m maximum wind gust, 10-minute window", **KMH_TO_MS,
        ),
        "pcpn_amt_pst1hr": cf.FieldSpec(
            var="precipitation_amount", standard_name="precipitation_amount",
            cell_methods="time: sum (interval: 1 hour)",
            long_name="Precipitation accumulated over the past hour", cf_units="kg m-2",
        ),
        "rnfl_amt_pst1hr": cf.FieldSpec(
            var="rainfall_amount", standard_name="thickness_of_rainfall_amount",
            cell_methods="time: sum (interval: 1 hour)",
            long_name="Rainfall accumulated over the past hour",
        ),
        "snw_dpth": cf.FieldSpec(
            var="surface_snow_thickness", standard_name="surface_snow_thickness",
            long_name="Snow depth", **CM_TO_M,
        ),
    }

    #: How far back to sample when enumerating stations.
    DISCOVERY_WINDOW = pd.Timedelta(hours=1)

    def discover_from_data(self, query: Query) -> list[StationMatch]:
        end = query.end
        start = max(query.start, end - self.DISCOVERY_WINDOW)
        params = {
            "bbox": ",".join(f"{v:.6f}" for v in (query.bbox or (-180, -90, 180, 90))),
            "datetime": (
                f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            ),
        }
        seen: dict[str, StationMatch] = {}
        for feature in paginate_ogc_items(
            self.items_url,
            params,
            provider=self.name,
            max_items=int(query.option("max_items", 50_000)),
        ):
            point = point_from_feature(feature)
            if point is None:
                continue
            lat, lon = point
            if not query.contains(lat, lon):
                continue
            props = feature.get("properties") or {}
            station_id = props.get("msc_id-value")
            if not station_id or str(station_id) in seen:
                continue
            match = self.new_match(
                station_id=str(station_id),
                name=str(props.get("stn_nam-value") or ""),
                lat=lat,
                lon=lon,
                variables=tuple(sorted(self.variables)),
                n_rows_est=int(query.days * self.samples_per_day),
                extra={"tc_id": props.get("tc_id-value")},
            )
            seen[str(station_id)] = match.attach_site(query)
        log.debug("eccc_swob discovered %d station(s)", len(seen))
        return list(seen.values())


# --------------------------------------------------------------------------- marine


#: The MSC Datamart, which is a different host from the provider's OGC API. Marine SWOB is
#: published here and nowhere else, so the URL lives beside the source that needs it rather
#: than on :class:`~omnisea.providers.eccc.EcccProvider`, whose ``base_url`` is the OGC API.
DATAMART_BASE = "https://dd.weather.gc.ca"

#: The date appears twice: once to pick the archive snapshot, once inside it. Both are the UTC
#: day of the observation.
_MARINE_DIR = "{stamp}/WXO-DD/observations/swob-ml/marine/moored-buoys/{stamp}/{buoy}"

#: ``2026-08-20-0005-4600146-AUTO-swob.xml`` — date, then HHMM, in an Apache autoindex page.
_SWOB_FILE = re.compile(r'href="((\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})-[^"/]*?\.xml)"')

_OM_NS = "{http://www.opengis.net/om/1.0}"
_PO_NS = "{http://dms.ec.gc.ca/schema/point-observation/2.0}"

#: SWOB-ML writes an absent reading as the literal string ``MSNG``. Left alone it would land in
#: the frame as text and turn an otherwise numeric column into an object column.
_MISSING = "MSNG"

_MARINE_SKIP = frozenset(
    {
        "dataset",
        "lat",
        "long",
        "stn_elev",
        "stn_nam",
        "stn_typ",
        "rpt_typ",
        "logr_typ",
        "wmo_id_extnd",
        "wmo_synop_id",
    }
)

_MEAN_10MIN = "time: mean (interval: 10 minutes)"
_MAX_10MIN = "time: maximum (interval: 10 minutes)"
_MEAN_20MIN = "time: mean (interval: 20 minutes)"
_MAX_20MIN = "time: maximum (interval: 20 minutes)"

_SPREAD_TYPO = (
    "ECCC's SWOB-ML guide labels this field wave direction spread but describes it as *wind* "
    "direction spread; the field name and the degrees-of-arc unit are taken as authoritative."
)


class EcccSwobMarine(_SwobSource):
    """Moored-buoy observations — waves, sea surface temperature and wind over water.

    The 42 buoys in ``swob-marine-stations`` report significant wave height, wave period and
    direction, sea surface temperature, wind, and pressure, hourly on the MSC-type buoys and
    every ten minutes on the newer OPP-type ones. The Datamart keeps about 30 days.

    Discovery reads the real catalogue, which is worth doing here in a way it is not for land
    SWOB: 42 stations of which 39 report today, and a station that has gone quiet costs one
    directory listing rather than a full data request.

    Retrieval is the unusual part. There is no query interface — each observation is its own
    XML file — so a window is assembled by listing one directory per station-day and pulling
    the files whose timestamps fall inside it. Listing rather than constructing the filenames
    is deliberate: the report minute is not fixed (MSC buoys file at ``HH05``, OPP buoys every
    ten minutes from ``HH00``), so guessed names would 404 on entire station classes.
    """

    name = "eccc_swob_marine"
    title = "ECCC marine surface weather observations (SWOB) — moored buoys"
    node_path = "in_situ/marine_buoy"
    collection = ""  # the observations are not an OGC collection; see the module docstring
    station_collection = "swob-marine-stations"
    catalogue_id_field = "msc_id"
    station_id_field = "msc_id"
    name_fields = ("name_en", "name_fr")
    time_field = "date_tm"
    skip_fields = _MARINE_SKIP
    #: The MSC-type buoys, which are the majority; OPP-type buoys report six times as often, so
    #: this is a floor on the row estimate rather than an average.
    samples_per_day = 24.0
    retention = pd.Timedelta(days=30)

    #: Default ceiling on observation files pulled per query. Lower than the OGC sources' ceiling
    #: because here the limit that bites is request count, not payload: one HTTP request buys one
    #: observation, so 30 days of a ten-minute buoy is already ~4,300 of them.
    DEFAULT_MAX_FILES = 10_000

    fields = {
        # ------------------------------------------------------------- atmosphere over water
        "avg_air_temp_pst10mts": cf.FieldSpec(
            var="air_temperature", standard_name="air_temperature",
            cell_methods=_MEAN_10MIN, long_name="Air temperature, 10-minute mean", **DEGC_TO_K,
        ),
        "avg_wnd_spd_pst10mts": cf.FieldSpec(
            var="wind_speed", standard_name="wind_speed",
            cell_methods=_MEAN_10MIN, long_name="Wind speed, 10-minute mean", **KMH_TO_MS,
        ),
        "max_wnd_spd_pst10mts": cf.FieldSpec(
            var="wind_speed_of_gust", standard_name="wind_speed_of_gust",
            cell_methods=_MAX_10MIN,
            long_name="Maximum wind speed, 10-minute window", **KMH_TO_MS,
        ),
        "avg_wnd_dir_pst10mts": cf.FieldSpec(
            var="wind_from_direction", standard_name="wind_from_direction",
            cell_methods=_MEAN_10MIN, long_name="Wind direction, 10-minute mean",
        ),
        "avg_stn_pres_pst10mts": cf.FieldSpec(
            var="air_pressure", standard_name="air_pressure",
            cell_methods=_MEAN_10MIN,
            long_name="Station air pressure, 10-minute mean", **HPA_TO_PA,
        ),
        "avg_mslp_pst10mts": cf.FieldSpec(
            var="air_pressure_at_mean_sea_level",
            standard_name="air_pressure_at_mean_sea_level",
            cell_methods=_MEAN_10MIN,
            long_name="Mean sea level pressure, 10-minute mean", **HPA_TO_PA,
        ),
        # A change *across* three hours, reported hourly, so its windows overlap. CF's
        # tendency_of_air_pressure is a rate in Pa s-1 and would misdescribe it; no cell_methods
        # either, since neither summing overlapping windows nor averaging them means anything.
        "pres_tend_amt_pst3hrs": cf.FieldSpec(
            var="pressure_change_past_3_hours", standard_name="",
            long_name="Air pressure change over the past 3 hours",
        ),
        # ------------------------------------------------------------- sea surface
        "avg_sea_sfc_temp_pst10mts": cf.FieldSpec(
            var="sea_surface_temperature", standard_name="sea_surface_temperature",
            cell_methods=_MEAN_10MIN,
            long_name="Sea surface temperature, 10-minute mean", **DEGC_TO_K,
        ),
        # ------------------------------------------------------------- waves, MSC-type buoys
        "avg_sig_wave_hgt_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_significant_height",
            standard_name="sea_surface_wave_significant_height",
            cell_methods=_MEAN_20MIN,
            long_name="Significant wave height, 20-minute record",
        ),
        "avg_sig_wave_pd_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_significant_period",
            standard_name="sea_surface_wave_significant_period",
            cell_methods=_MEAN_20MIN,
            long_name="Significant wave period, 20-minute record",
        ),
        "spetrl_sig_wave_hgt_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_significant_height_from_spectrum",
            standard_name="sea_surface_wave_significant_height",
            long_name="Spectral significant wave height, 20-minute record",
            comment=(
                "Hm0, estimated from the variance spectrum. Carries the same standard name as "
                "the time-domain estimate but is a separate measurement, so it travels as its "
                "own variable rather than being merged with it."
            ),
        ),
        "spetrl_wave_enrgy_pd_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_energy_period",
            standard_name=(
                "sea_surface_wave_mean_period_from_variance_spectral_density"
                "_inverse_frequency_moment"
            ),
            long_name="Spectral wave energy period, 20-minute record",
        ),
        # ECCC documents this only as "average spectral wave period" and does not say which
        # frequency moment it comes from, and CF has a distinct name per moment. Naming one
        # would be a guess, so the quantity travels described but not standardised.
        "avg_spetrl_wave_pd_pst20mts": cf.FieldSpec(
            var="spectral_mean_wave_period", standard_name="",
            cell_methods=_MEAN_20MIN,
            long_name="Spectral mean wave period, 20-minute record",
        ),
        "avg_wave_hgt_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_mean_height",
            standard_name="sea_surface_wave_mean_height",
            cell_methods=_MEAN_20MIN, long_name="Mean wave height, 20-minute record",
        ),
        "avg_wave_pd_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_mean_period",
            standard_name="sea_surface_wave_mean_period",
            cell_methods=_MEAN_20MIN, long_name="Mean wave period, 20-minute record",
        ),
        "max_wave_hgt_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_maximum_height",
            standard_name="sea_surface_wave_maximum_height",
            cell_methods=_MAX_20MIN, long_name="Maximum wave height, 20-minute record",
        ),
        "max_wave_crst_hgt_abv_avg_wtr_lvl_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_maximum_crest_height",
            standard_name="sea_surface_wave_maximum_crest_height",
            cell_methods=_MAX_20MIN,
            long_name="Maximum wave crest height above mean water level, 20-minute record",
        ),
        # The period belonging to one event, not a reduction over time: taking the maximum of
        # these when resampling would report a period no wave actually had.
        "pd_of_max_wave_hgt_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_period_of_highest_wave",
            standard_name="sea_surface_wave_period_of_highest_wave",
            long_name="Period of the highest wave, 20-minute record",
        ),
        "pk_wave_pd_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_period_at_variance_spectral_density_maximum",
            standard_name="sea_surface_wave_period_at_variance_spectral_density_maximum",
            long_name="Peak wave period, 20-minute record",
        ),
        # Wave directions follow the meteorological convention, like the wind: measured against
        # 103 samples of developed wind sea (>=25 km/h wind, >=1 m Hs), the wave direction sits
        # a circular-mean 12 degrees off the wind's *from* direction, not 180 degrees off it.
        "avg_wave_dir_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_from_direction",
            standard_name="sea_surface_wave_from_direction",
            cell_methods=_MEAN_20MIN, long_name="Mean wave direction, 20-minute record",
        ),
        "avg_pk_wave_dir_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_from_direction_at_variance_spectral_density_maximum",
            standard_name="sea_surface_wave_from_direction_at_variance_spectral_density_maximum",
            cell_methods=_MEAN_20MIN, long_name="Peak wave direction, 20-minute record",
        ),
        "avg_wave_dir_sprd_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_directional_spread",
            standard_name="sea_surface_wave_directional_spread",
            cell_methods=_MEAN_20MIN,
            long_name="Wave directional spread, 20-minute record", comment=_SPREAD_TYPO,
        ),
        "pk_wave_dir_sprd_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_directional_spread_at_variance_spectral_density_maximum",
            standard_name=(
                "sea_surface_wave_directional_spread_at_variance_spectral_density_maximum"
            ),
            long_name="Wave directional spread at the spectral peak, 20-minute record",
            comment=_SPREAD_TYPO,
        ),
        # ------------------------------------------------------------- waves, OPP-type buoys
        # Same quantities under the newer buoys' spellings. They never co-occur with the names
        # above on one buoy, so the two blocks describe alternatives, not duplicates.
        "sig_wave_hgt_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_significant_height",
            standard_name="sea_surface_wave_significant_height",
            long_name="Significant wave height, 20-minute record",
        ),
        "sig_wave_pd_pst20mts": cf.FieldSpec(
            var="sea_surface_wave_significant_period",
            standard_name="sea_surface_wave_significant_period",
            long_name="Significant wave period, 20-minute record",
        ),
    }

    # ------------------------------------------------------------------ discovery

    def station_from_feature(
        self, query: Query, feature: Mapping[str, Any]
    ) -> StationMatch | None:
        """One buoy from the marine catalogue, carrying the id the Datamart files it under.

        A buoy with no usable WMO id is dropped rather than kept: its directory name cannot be
        formed, so it would appear in the catalogue and then return nothing.
        """
        match = super().station_from_feature(query, feature)
        if match is None:
            return None
        props = feature.get("properties") or {}
        buoy_id = _extended_wmo_id(props.get("wmo_id"))
        if not buoy_id:
            log.debug("%s: buoy %s has no wmo_id; skipping", self.name, match.station_id)
            return None
        match.extra["buoy_id"] = buoy_id
        return match

    # ------------------------------------------------------------------ retrieval

    def fetch(self, query: Query, matches: list[StationMatch]) -> list[StationSeries]:
        """Assemble every station's window from individual Datamart files.

        Done in two flat passes — list all the directories, then pull all the files — rather
        than a pool per station. Nesting pools would put ``max_workers`` squared requests in
        flight at the Datamart, which does not have the OGC API's shared concurrency limiter
        in front of it.
        """
        if not matches:
            return []
        workers = int(query.option("max_workers", DEFAULT_MAX_WORKERS))
        ceiling = int(query.option("max_items", self.DEFAULT_MAX_FILES))

        days = [(match, stamp) for match in matches for stamp in _utc_days(query)]
        listings = map_threads(
            lambda job: self._list_day(query, *job),
            days,
            max_workers=workers,
            label=f"{self.name} day",
        )

        jobs: list[tuple[str, str]] = []  # (station_id, observation url)
        for (match, _stamp), urls in zip(days, listings, strict=True):
            jobs.extend((match.station_id, url) for url in urls)

        if len(jobs) > ceiling:
            raise PayloadTooLargeError(
                f"{self.name} would need {len(jobs):,} requests for this window, over the "
                f"{ceiling:,} ceiling — the Datamart publishes one file per observation, so "
                "the cost is in round trips. Narrow the time window or the area, or raise "
                "max_items.",
                estimate=len(jobs),
                limit=ceiling,
            )

        documents = map_threads(
            lambda job: _get_text(job[1], source=self.name),
            jobs,
            max_workers=workers,
            label=f"{self.name} observation",
        )

        by_station: dict[str, list[dict[str, Any]]] = {m.station_id: [] for m in matches}
        for (station_id, url), text in zip(jobs, documents, strict=True):
            if not text:
                continue
            row = _parse_swob_ml(text)
            if row is None:
                log.debug("%s: unreadable observation at %s", self.name, url)
                continue
            by_station[station_id].append(row)

        out: list[StationSeries] = []
        for match in matches:
            series = self.series_from_rows(query, match, by_station[match.station_id])
            if series is not None:
                out.append(series)
        return out

    def _list_day(self, query: Query, match: StationMatch, stamp: str) -> list[str]:
        """Observation URLs for one buoy on one UTC day, narrowed to the query window.

        The first and last day of a window are usually only partly inside it, and each file
        outside it would be a round trip spent on a row ``trim_to_window`` then discards. The
        filename carries the report time, so they can be dropped before anything is fetched.
        """
        path = _MARINE_DIR.format(stamp=stamp, buoy=match.require("buoy_id"))
        directory = f"{DATAMART_BASE}/{path}/"
        listing = _get_text(directory, source=self.name)
        if listing is None:
            return []  # buoy not deployed that day, or the day has aged out of the archive
        return [
            directory + name
            for name, reported in _observation_files(listing)
            if query.start <= reported <= query.end
        ]

    def node_attrs(self, query: Query, match: StationMatch) -> dict[str, Any]:
        return self.base_attrs(
            title=f"{match.name} ({match.station_id}) — {self.title or self.name}",
            source_url=(
                f"{DATAMART_BASE}/today/observations/swob-ml/marine/moored-buoys/"
                f"?buoy={match.extra.get('buoy_id', '')}"
            ),
            collection=self.station_collection,
            station_id=match.station_id,
            site=match.site,
        )


# --------------------------------------------------------------------------- marine helpers


def _extended_wmo_id(wmo_id: Any) -> str:
    """The Datamart's directory name for a buoy, from the catalogue's WMO id.

    The catalogue publishes both forms — five digits for the older buoys (``46146``) and seven
    for the newer ones (``4600303``) — while the Datamart only ever uses the seven-digit one.
    The five-digit form expands by re-inserting the two zeros it drops.
    """
    text = str(wmo_id or "").strip()
    if not text.isdigit():
        return ""
    if len(text) == 5:
        return f"{text[:2]}00{text[2:]}"
    return text if len(text) == 7 else ""


def _utc_days(query: Query) -> list[str]:
    """Every UTC day the window touches, as ``YYYYMMDD``."""
    start = query.start.tz_convert("UTC").normalize()
    end = query.end.tz_convert("UTC").normalize()
    return [d.strftime("%Y%m%d") for d in pd.date_range(start, end, freq="D", tz="UTC")]


def _observation_files(listing: str) -> list[tuple[str, pd.Timestamp]]:
    """``(filename, report time)`` for every SWOB-ML file on an Apache autoindex page.

    The report time is read from the filename rather than the file, so a window can be narrowed
    before anything is downloaded. The name is only a label — the row's own ``date_tm`` is what
    ends up in the index.
    """
    out: list[tuple[str, pd.Timestamp]] = []
    for match in _SWOB_FILE.finditer(listing):
        name, day, hour, minute = match.groups()
        try:
            reported = pd.Timestamp(f"{day}T{hour}:{minute}", tz="UTC")
        except ValueError:  # a filename that looks the part but is not a real instant
            continue
        out.append((name, reported))
    return out


def _get_text(url: str, *, source: str) -> str | None:
    """GET a Datamart document, or ``None`` when it is simply not there.

    ``http.get_json`` cannot be reused here: the Datamart serves XML observations and HTML
    directory listings, and a 404 is ordinary rather than exceptional — a buoy that was not
    deployed on a given day has no directory for it. Everything else still surfaces as an
    :class:`~omnisea.errors.UpstreamError`, so a real outage is not mistaken for a gap.
    """
    session = get_session()
    log.debug("GET %s", url)
    try:
        resp = session.get(url, timeout=DEFAULT_TIMEOUT, headers={"Accept": "*/*"})
    except requests.RequestException as exc:
        raise UpstreamError(f"request to {url} failed: {exc}", provider=source, url=url) from exc
    if resp.status_code == 404:
        return None
    if not resp.ok:
        raise UpstreamError(
            "upstream request failed",
            provider=source,
            url=resp.url,
            status=resp.status_code,
        )
    # Trust only a charset the server actually declares. `requests` otherwise falls back to
    # ISO-8859-1 for any text/* body, and SWOB-ML is UTF-8 with a degree sign in the `-uom` of
    # every temperature and direction — decoded as Latin-1 the units silently become mojibake.
    if "charset=" not in resp.headers.get("Content-Type", "").lower():
        resp.encoding = "utf-8"
    return resp.text


def _parse_swob_ml(text: str) -> dict[str, Any] | None:
    """One SWOB-ML observation, flattened into the shape the GeoJSON collections publish.

    Every measurement becomes three keys — the value, a ``-uom`` sibling and a ``-qa`` sibling —
    so a marine observation is indistinguishable from a ``swob-realtime`` feature's properties
    downstream, and the unit-from-the-data rule keeps working unchanged.
    """
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return None
    observation = root.find(f"{_OM_NS}member/{_OM_NS}Observation")
    if observation is None:
        return None

    row: dict[str, Any] = {}
    ident_path = f"{_OM_NS}metadata/{_PO_NS}set/{_PO_NS}identification-elements"
    _collect(observation.find(ident_path), row)
    _collect(observation.find(f"{_OM_NS}result/{_PO_NS}elements"), row)
    return row or None


def _collect(container: Iterable[Any] | None, row: dict[str, Any]) -> None:
    if container is None:
        return
    for element in container:
        name = element.get("name")
        if not name:
            continue
        row[name] = _value(element.get("value"))
        uom = element.get("uom")
        if uom:
            row[f"{name}-uom"] = uom
        for qualifier in element:
            if qualifier.get("name") == "qa_summary":
                row[f"{name}-qa"] = _value(qualifier.get("value"))


def _value(raw: Any) -> Any:
    """A reading as a number where it is one, and ``None`` where SWOB says it is absent."""
    if raw is None or raw == "" or raw == _MISSING:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return raw
