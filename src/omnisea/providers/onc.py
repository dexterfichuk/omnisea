"""Ocean Networks Canada — the Oceans 3.0 API.

``https://data.oceannetworks.ca/api`` (`OpenAPI <https://data.oceannetworks.ca/OpenAPI>`_)

ONC runs cabled observatories, moorings and autonomous platforms off both Canadian coasts and
in the Arctic: 1,992 locations, 219 measured properties, decades of record. It is the one
built-in source that **requires a credential**, which shapes most of what follows.

Getting a token
---------------

Register at https://data.oceannetworks.ca, then Profile -> Web Services API. Pass it either way::

    omnisea.fetch(..., onc_token="…")     # explicit
    export ONC_TOKEN=…                     # or the environment, picked up automatically

**The token is a secret in a query string.** ONC authenticates with ``?token=`` rather than a
header, and a URL like that would otherwise reach three places that outlive the request: the
debug log, the message of every :class:`~omnisea.errors.UpstreamError`, and the ``source_url``
recorded on every node — which is written into netCDF files people share and commit. So
:func:`omnisea.http.get_json` redacts it from all three (see
:data:`omnisea.http.SENSITIVE_PARAMS`), and the ``source_url`` this module records is built
without the token rather than sanitized afterwards.

Four upstream facts shape this adapter, each verified live
----------------------------------------------------------

* **There is no spatial filter on ``/api/locations``.** No ``lat``/``lon``/``radius``, no bbox —
  the parameters are rejected by name. The full list is 1,992 locations in one ~830 KB payload,
  so it is fetched once per process and filtered client-side, exactly as DFO's IWLS station list
  is. ``propertyCode`` *is* accepted, and narrows the list server-side when the caller asked for
  a specific variable.
* **Scalar data is columnar, not row-oriented.** A response is one object per sensor holding
  three parallel arrays — ``values``, ``sampleTimes``, ``qaqcFlags``. Every other source here
  returns rows, so this is the one adapter that transposes.
* **Sampling can be 1 Hz.** Folger Deep's CTD returns a value every second: a single day is
  86,400 rows per sensor, and eight sensors makes it 691,200. ONC offers server-side
  ``resamplePeriod``, so omnisea asks for a sane default rather than pulling raw seconds and
  throwing them away locally.
* **Every response carries its own citation, with a DOI.** ONC returns the exact wording it
  wants credited plus a resolvable DOI. That is better provenance than omnisea could assemble
  itself, so it is recorded on the node and flows into :func:`omnisea.citation`.

Why not the ``onc`` package
---------------------------

ONC publishes an official Python client. It would work, and it is the same trade-off already
made for ERDDAP: the easy half is building these URLs, and the hard half is routing them through
:mod:`omnisea.http` so the retry ladder, the global concurrency cap, the User-Agent, the payload
ceiling and the token redaction all apply. A client with its own transport sits outside every one
of those. The URLs here are a few lines; the safety is not.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import pandas as pd

from .. import cf
from ..errors import ProviderError, QueryError, UpstreamError
from ..http import DEFAULT_MAX_WORKERS, NEVER_CACHE, get_json, map_threads
from ..query import Query, register_option
from .base import (
    Provider,
    RetrievalSource,
    StationMatch,
    StationSeries,
    drop_orphan_qc,
    trim_to_window,
)

log = logging.getLogger("omnisea.onc")

__all__ = ["OncProvider", "OncScalarDataSource", "clear_cache", "PROPERTY_FIELDS"]

BASE = "https://data.oceannetworks.ca/api"

register_option("onc_token", "onc: Oceans 3.0 API token (or set the ONC_TOKEN environment var)")
register_option(
    "onc_resample_seconds",
    "onc: server-side resampling interval in seconds; None or 0 for raw samples (default 60)",
)
register_option(
    "onc_device_categories",
    "onc: device category code(s) to pull, e.g. 'CTD'; default is every category at the location",
)

#: ONC samples as fast as 1 Hz, so a raw day of one CTD is 86,400 rows per sensor. A minute is
#: fine enough for anything omnisea's ``align()`` is likely to join against and three orders of
#: magnitude smaller. Ask for raw data explicitly with ``onc_resample_seconds=0``.
DEFAULT_RESAMPLE_SECONDS = 60

#: ONC's own flag vocabulary, from the ``qaqcFlagInfo`` block every response carries.
QAQC_FLAGS = {
    0: "No Quality Control",
    1: "Data Passed All Tests",
    2: "Data Probably Good",
    3: "Data Probably Bad",
    4: "Data Bad",
    6: "Insufficient Valid Data for Reliable Down-Sampling (ONC-defined flag)",
    7: "Averaged Value (ONC defined flag)",
    8: "Interpolated Value",
    9: "Missing Data",
}

#: ONC ``propertyCode`` -> CF description, for the properties an ocean scientist reaches for
#: first. This is a **floor, not an inventory**: ONC publishes 219 properties and every one of
#: them is returned, with the ones named here carrying a CF ``standard_name`` and the rest
#: travelling under ONC's own code, tagged ``omnisea_mapped = 0``.
#:
#: Units are deliberately left ``None`` and read from each sensor's ``unitOfMeasure`` at fetch
#: time: the same property is served in different units by different instruments, and a
#: hardcoded table would eventually disagree with the numbers beside it.
PROPERTY_FIELDS: dict[str, cf.FieldSpec] = {
    "seawatertemperature": cf.FieldSpec(
        var="sea_water_temperature", standard_name="sea_water_temperature", units=None,
        long_name="Sea water temperature",
    ),
    "salinity": cf.FieldSpec(
        var="sea_water_practical_salinity", standard_name="sea_water_practical_salinity",
        units=None, long_name="Practical salinity",
    ),
    "conductivity": cf.FieldSpec(
        var="sea_water_electrical_conductivity",
        standard_name="sea_water_electrical_conductivity", units=None,
        long_name="Sea water electrical conductivity",
    ),
    "density": cf.FieldSpec(
        var="sea_water_density", standard_name="sea_water_density", units=None,
        long_name="Sea water density",
    ),
    "pressure": cf.FieldSpec(
        var="sea_water_pressure", standard_name="sea_water_pressure", units=None,
        long_name="Sea water pressure",
    ),
    "depth": cf.FieldSpec(
        var="depth", standard_name="depth", units=None, long_name="Depth below sea surface",
    ),
    "oxygen": cf.FieldSpec(
        var="mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
        standard_name="mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
        units=None, long_name="Dissolved oxygen concentration",
    ),
    "oxygensaturation": cf.FieldSpec(
        var="fractional_saturation_of_oxygen_in_sea_water",
        standard_name="fractional_saturation_of_oxygen_in_sea_water", units=None,
        long_name="Oxygen saturation",
    ),
    "chlorophyll": cf.FieldSpec(
        var="mass_concentration_of_chlorophyll_in_sea_water",
        standard_name="mass_concentration_of_chlorophyll_in_sea_water", units=None,
        long_name="Chlorophyll concentration",
    ),
    "turbidityntu": cf.FieldSpec(
        var="sea_water_turbidity", standard_name="sea_water_turbidity", units=None,
        long_name="Turbidity",
    ),
    "ph": cf.FieldSpec(
        var="sea_water_ph_reported_on_total_scale",
        standard_name="sea_water_ph_reported_on_total_scale", units=None,
        long_name="pH (total scale)",
    ),
    "sigmatheta": cf.FieldSpec(
        var="sea_water_sigma_theta", standard_name="sea_water_sigma_theta", units=None,
        long_name="Sigma-theta",
    ),
    "soundspeed": cf.FieldSpec(
        var="speed_of_sound_in_sea_water", standard_name="speed_of_sound_in_sea_water",
        units=None, long_name="Speed of sound in sea water",
    ),
    "seafloorpressure": cf.FieldSpec(
        var="sea_water_pressure_at_sea_floor",
        standard_name="sea_water_pressure_at_sea_floor", units=None,
        long_name="Pressure at the sea floor",
    ),
    "absolutebarometricpressure": cf.FieldSpec(
        var="air_pressure", standard_name="air_pressure", units=None,
        long_name="Absolute barometric pressure",
    ),
    "airtemperature": cf.FieldSpec(
        var="air_temperature", standard_name="air_temperature", units=None,
        long_name="Air temperature",
    ),
    "relativehumidity": cf.FieldSpec(
        var="relative_humidity", standard_name="relative_humidity", units=None,
        long_name="Relative humidity",
    ),
    "windspeed": cf.FieldSpec(
        var="wind_speed", standard_name="wind_speed", units=None, long_name="Wind speed",
    ),
    "winddirection": cf.FieldSpec(
        var="wind_from_direction", standard_name="wind_from_direction", units=None,
        long_name="Wind direction (from)",
    ),
}

_locations_cache: list[dict[str, Any]] | None = None
_property_locations_cache: dict[str, list[dict[str, Any]]] = {}
_categories_cache: dict[str, list[dict[str, Any]]] = {}
_lock = threading.Lock()


def clear_cache() -> None:
    """Drop the cached location list, per-property lists and device categories (used by tests)."""
    global _locations_cache
    with _lock:
        _locations_cache = None
        _property_locations_cache.clear()
        _categories_cache.clear()


class OncProvider(Provider):
    """Ocean Networks Canada, Oceans 3.0."""

    name = "onc"
    title = "Ocean Networks Canada"
    base_url = BASE
    license = "Ocean Networks Canada — data are licensed per dataset; cite the DOI on each node"
    terms_url = "https://www.oceannetworks.ca/data-tools/data-policy/"

    #: The catalogue endpoints are near-static — locations, properties and device categories
    #: change when infrastructure is deployed or retired, not between queries. ``scalardata``
    #: is the measurement endpoint and is excluded outright, listed first so nothing else can
    #: claim it. Note these patterns match the URL *including* its query string, and the token
    #: is part of that; requests-cache keys on the whole URL, so two different tokens simply do
    #: not share cache entries, which is the correct behaviour for a per-user credential.
    cache_policy = {
        "data.oceannetworks.ca/api/scalardata*": NEVER_CACHE,
        "data.oceannetworks.ca/api/rawdata*": NEVER_CACHE,
        "data.oceannetworks.ca/api/locations*": timedelta(days=7),
        "data.oceannetworks.ca/api/properties*": timedelta(days=7),
        "data.oceannetworks.ca/api/deviceCategories*": timedelta(days=7),
    }

    def clear_cache(self) -> None:
        clear_cache()

    def build_sources(self) -> Sequence[RetrievalSource]:
        return [OncScalarDataSource(self)]

    # ------------------------------------------------------------------ auth

    def has_token(self, query: Query) -> bool:
        """Is a credential available at all? Used to opt out quietly rather than fail loudly."""
        return bool(query.option("onc_token") or os.environ.get("ONC_TOKEN"))

    def token(self, query: Query) -> str:
        """The API token, from the query or the environment, or a message saying how to get one.

        Never logged and never recorded on a node — see the module docstring.
        """
        token = query.option("onc_token") or os.environ.get("ONC_TOKEN")
        if not token:
            raise ProviderError(
                "Ocean Networks Canada requires an API token. Register at "
                "https://data.oceannetworks.ca, then Profile -> Web Services API, and pass it "
                "as omnisea.fetch(..., onc_token='...') or set the ONC_TOKEN environment "
                "variable. Every other omnisea source works without one.",
                provider="onc",
            )
        return str(token)

    def api(self, path: str, params: dict[str, Any], *, query: Query, source: str) -> Any:
        """GET an ONC endpoint with the token attached.

        The token goes in ``params`` rather than the URL string so that it is redacted
        everywhere a failure is reported: :func:`omnisea.http.get_json` scrubs it from the log
        line, and from the URL carried on any :class:`~omnisea.errors.UpstreamError`.
        """
        return get_json(
            f"{BASE}/{path.lstrip('/')}",
            {**params, "token": self.token(query)},
            provider=source,
        )

    # ------------------------------------------------------------------ catalogue

    def locations(self, query: Query, *, property_code: str | None = None) -> list[dict[str, Any]]:
        """Every ONC location, or those serving one property. Fetched once per process.

        ``/api/locations`` has no spatial filter at all — ``lat``, ``lon`` and ``radius`` are
        rejected by name — so the whole list comes down and omnisea filters it. ``propertyCode``
        is accepted, and cuts 1,992 locations to the few hundred that measure the thing asked
        for, which is worth a separate cached list.
        """
        global _locations_cache
        if property_code:
            with _lock:
                cached = _property_locations_cache.get(property_code)
            if cached is not None:
                return cached
            found = self.api(
                "locations", {"method": "get", "propertyCode": property_code},
                query=query, source="onc_scalardata",
            )
            rows = found if isinstance(found, list) else []
            with _lock:
                _property_locations_cache[property_code] = rows
            return rows

        with _lock:
            if _locations_cache is not None:
                return _locations_cache
        log.debug("fetching the ONC location list")
        found = self.api("locations", {"method": "get"}, query=query, source="onc_scalardata")
        rows = found if isinstance(found, list) else []
        with _lock:
            _locations_cache = rows
        return rows

    def device_categories(self, query: Query, location_code: str) -> list[dict[str, Any]]:
        """Device categories deployed at one location, memoized.

        ``scalardata`` needs a ``deviceCategoryCode``; a location has several (Folger Deep has
        six), and there is no "everything here" wildcard, so they are enumerated first.
        """
        with _lock:
            cached = _categories_cache.get(location_code)
        if cached is not None:
            return cached
        try:
            found = self.api(
                "deviceCategories", {"method": "get", "locationCode": location_code},
                query=query, source="onc_scalardata",
            )
        except Exception:  # noqa: BLE001 - a location with no categories is not a failure
            log.debug("no device categories for %s", location_code, exc_info=True)
            found = []
        rows = found if isinstance(found, list) else []
        with _lock:
            _categories_cache[location_code] = rows
        return rows


class OncScalarDataSource(RetrievalSource):
    """Scalar sensor time series from any ONC location.

    One node per (location, device category), because a location can carry a CTD and a
    hydrophone and a current meter, and they are different instruments with different cadences
    that should not be forced onto one time index.
    """

    name = "onc_scalardata"
    title = "Ocean Networks Canada scalar sensor data"
    node_path = "in_situ/onc"
    feature_type = "timeSeries"
    fields = PROPERTY_FIELDS
    #: Sensor units are read from each response's ``unitOfMeasure`` rather than declared, so an
    #: empty ``units`` on a FieldSpec above is deliberate, not an omission.
    fields_from_metadata = True

    # ------------------------------------------------------------------ discovery

    def discover(self, query: Query) -> list[StationMatch]:
        if not self.wants_anything(query):
            return []

        # Without a credential this source simply cannot participate. Raising would print a
        # failure on *every* omnisea call made by the majority of users who have no ONC token —
        # a permanent banner for something they never asked for. Naming ONC explicitly is
        # reserved for callers who asked for it, where silence really would read as "ONC has
        # nothing near you".
        asked_for_onc = bool(query.providers) and any(
            name in ("onc", self.name) for name in query.providers
        )
        if not self.provider.has_token(query):
            if asked_for_onc:
                self.provider.token(query)  # raises, with the how-to-get-one message
            log.debug("%s: no ONC token configured; skipping", self.name)
            return []

        locations = self._candidate_locations(query)
        matches: list[StationMatch] = []
        for location in locations:
            lat, lon = location.get("lat"), location.get("lon")
            if lat is None or lon is None:
                continue
            if not query.contains(float(lat), float(lon)):
                continue
            if str(location.get("hasDeviceData")).lower() != "true":
                # Nothing has ever been deployed here; scalardata would return an error.
                continue
            match = self.new_match(
                station_id=str(location.get("locationCode")),
                name=str(location.get("locationName") or ""),
                lat=float(lat),
                lon=float(lon),
                variables=tuple(sorted(self.variables)),
                n_rows_est=self._estimate_rows(query),
                extra={
                    "location_code": location.get("locationCode"),
                    "depth": location.get("depth"),
                    "description": location.get("description"),
                    "data_search_url": location.get("dataSearchURL"),
                },
            )
            matches.append(match.attach_site(query))
        log.debug("onc_scalardata discovered %d location(s)", len(matches))
        return matches

    def _candidate_locations(self, query: Query) -> list[dict[str, Any]]:
        """Locations worth considering: narrowed server-side by property where we can.

        Asking for one variable turns 1,992 candidates into the few hundred that measure it,
        and ONC does that filtering itself. With several requested the lists are unioned, and
        with none the whole catalogue is scanned.
        """
        codes = self._requested_property_codes(query)
        if not codes:
            return self.provider.locations(query)
        seen: dict[str, dict[str, Any]] = {}
        for code in sorted(codes):
            for location in self.provider.locations(query, property_code=code):
                key = str(location.get("locationCode"))
                seen.setdefault(key, location)
        return list(seen.values())

    def _requested_property_codes(self, query: Query) -> set[str]:
        """ONC property codes matching the caller's ``variables=``, if any are recognisable."""
        if query.variables is None:
            return set()
        wanted = cf.resolve_names(query.variables) or frozenset()
        codes = {
            code
            for code, spec in PROPERTY_FIELDS.items()
            if code in wanted or spec.var in wanted
            or (spec.standard_name and spec.standard_name in wanted)
        }
        # A name omnisea does not curate might still be an ONC property code spelled exactly.
        codes |= {str(name) for name in wanted if str(name).islower() and str(name).isalpha()}
        return codes

    def _estimate_rows(self, query: Query) -> int:
        seconds = self._resample_seconds(query)
        per_day = 86_400 / seconds if seconds else 86_400.0
        return int(query.days * per_day)

    @staticmethod
    def _resample_seconds(query: Query) -> int:
        raw = query.option("onc_resample_seconds", DEFAULT_RESAMPLE_SECONDS)
        if raw in (None, 0, "0"):
            return 0
        try:
            seconds = int(raw)
        except (TypeError, ValueError):
            raise QueryError(
                f"onc_resample_seconds must be a whole number of seconds; got {raw!r}"
            ) from None
        if seconds < 0:
            raise QueryError(f"onc_resample_seconds must not be negative; got {seconds}")
        return seconds

    # ------------------------------------------------------------------ retrieval

    def fetch(self, query: Query, matches: list[StationMatch]) -> list[StationSeries]:
        jobs: list[tuple[StationMatch, str]] = []
        for match in matches:
            for category in self._categories_for(query, match):
                jobs.append((match, category))
        if not jobs:
            return []
        results = map_threads(
            lambda job: self._fetch_one(query, *job),
            jobs,
            max_workers=int(query.option("max_workers", DEFAULT_MAX_WORKERS)),
            label=f"{self.name} location/category",
        )
        return [series for series in results if series is not None]

    def _categories_for(self, query: Query, match: StationMatch) -> list[str]:
        """Which device categories to pull at this location — only ones deployed there.

        A requested category is intersected with what the location actually has rather than
        sent blindly: ONC answers a category that was never deployed with an HTTP 400, so
        asking for ``CTD`` across a radius would fail the whole query on the first location
        that happens to be a seismometer.
        """
        available = [
            str(c["deviceCategoryCode"])
            for c in self.provider.device_categories(query, match.require("location_code"))
            if c.get("deviceCategoryCode")
        ]
        requested = query.option("onc_device_categories")
        if not requested:
            return available
        wanted = [requested] if isinstance(requested, str) else [str(c) for c in requested]
        if not available:
            # The category list could not be read; trust the caller rather than skipping.
            return wanted
        chosen = [c for c in wanted if c in available]
        for missing in (c for c in wanted if c not in available):
            log.debug(
                "%s: %s has no %s deployment; skipping it there",
                self.name, match.station_id, missing,
            )
        return chosen

    def _fetch_one(
        self, query: Query, match: StationMatch, category: str
    ) -> StationSeries | None:
        location_code = match.require("location_code")
        params: dict[str, Any] = {
            "method": "getByLocation",
            "locationCode": location_code,
            "deviceCategoryCode": category,
            "dateFrom": _iso(query.start),
            "dateTo": _iso(query.end),
        }
        seconds = self._resample_seconds(query)
        if seconds:
            params["resamplePeriod"] = seconds

        try:
            payload = self.provider.api(
                "scalardata", params, query=query, source=self.name
            )
        except UpstreamError as exc:
            if _means_no_data(exc):
                log.debug(
                    "%s: no %s data at %s in this window", self.name, category, location_code
                )
                return None
            raise
        if not isinstance(payload, dict):
            return None
        sensors = payload.get("sensorData") or []
        if not sensors:
            return None

        frame, var_attrs = self._shape(query, sensors)
        return StationSeries(
            match=match,
            frame=frame,
            node_path=f"{self.node_path}/{_safe(location_code)}/{_safe(category)}",
            attrs=self._node_attrs(query, match, category, payload, seconds),
            var_attrs=var_attrs,
        )

    def _shape(
        self, query: Query, sensors: list[dict[str, Any]]
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        """Transpose ONC's columnar sensors into one time-indexed frame.

        Each sensor carries three parallel arrays rather than a list of rows, and different
        sensors on one instrument can have different lengths, so they are built as individual
        series and joined on time rather than assumed to line up.
        """
        to_cf = self.to_cf_units(query)
        include_unmapped = self.include_unmapped(query)

        pieces: list[pd.Series] = []
        var_attrs: dict[str, dict[str, Any]] = {}
        taken: set[str] = set()

        for sensor in sensors:
            property_code = str(sensor.get("propertyCode") or "")
            spec = PROPERTY_FIELDS.get(property_code)
            if spec is None:
                if not include_unmapped:
                    continue
                spec = cf.passthrough_spec(
                    property_code or str(sensor.get("sensorCode") or "sensor")
                )
            units = sensor.get("unitOfMeasure")
            if units:
                # Units always come from the sensor: ONC serves the same property in different
                # units from different instruments, so the table must not overrule the data.
                spec = cf.FieldSpec(**{**spec.__dict__, "units": str(units)})

            name = spec.var
            if name in taken:
                # Two sensors of the same property on one instrument — a redundant CTD pair, or
                # the same probe at two depths. Both are real; neither may overwrite the other.
                suffix = 2
                while f"{name}_{suffix}" in taken:
                    suffix += 1
                name = f"{name}_{suffix}"
            taken.add(name)

            data = sensor.get("data") or {}
            times = data.get("sampleTimes") or []
            values = data.get("values") or []
            flags = data.get("qaqcFlags") or []
            if not times:
                continue

            index = pd.to_datetime(pd.Series(times), utc=True, format="mixed", errors="coerce")
            converted = [cf.convert(v, spec, to_cf_units=to_cf) for v in values]
            series = pd.Series(converted[: len(index)], index=index[: len(converted)], name=name)
            series = series[series.index.notna()]
            series = series[~series.index.duplicated(keep="last")]
            pieces.append(series)
            var_attrs[name] = cf.cf_attrs(spec, units=spec.units, to_cf_units=to_cf)
            var_attrs[name].update({
                "onc_property_code": property_code,
                "onc_sensor_code": str(sensor.get("sensorCode") or ""),
                "onc_sensor_name": str(sensor.get("sensorName") or ""),
            })

            if flags:
                qc = pd.Series(
                    list(flags)[: len(index)], index=index[: len(flags)], name=f"{name}_qc"
                )
                qc = qc[qc.index.notna()]
                qc = qc[~qc.index.duplicated(keep="last")]
                pieces.append(qc)
                var_attrs[f"{name}_qc"] = {
                    "long_name": f"ONC QAQC flag for {name}",
                    "comment": "; ".join(f"{k}={v}" for k, v in sorted(QAQC_FLAGS.items())),
                    "source_field": "qaqcFlags",
                    cf.MAPPED_ATTR: 0,
                }

        if not pieces:
            return pd.DataFrame(), {}

        frame = pd.concat(pieces, axis=1).sort_index()
        frame.index.name = "time"
        frame = trim_to_window(frame, query.start, query.end)
        frame = frame.dropna(axis=1, how="all")
        frame = drop_orphan_qc(frame)
        var_attrs = {k: v for k, v in var_attrs.items() if k in frame.columns}
        return frame, var_attrs

    def _node_attrs(
        self,
        query: Query,
        match: StationMatch,
        category: str,
        payload: dict[str, Any],
        seconds: int,
    ) -> dict[str, Any]:
        """Node attributes, including ONC's own citation and DOI.

        ONC returns the exact wording it wants credited and a resolvable DOI per deployment.
        That is better attribution than omnisea could assemble from the query, so it is carried
        through to :func:`omnisea.citation` rather than replaced.
        """
        citations = payload.get("citations") or []
        attrs = self.base_attrs(
            title=f"{match.name} ({match.station_id}) — ONC {category}",
            # Deliberately without the token: this string is stamped on the node and written
            # into netCDF files that get shared.
            source_url=(
                f"{BASE}/scalardata?method=getByLocation"
                f"&locationCode={match.station_id}&deviceCategoryCode={category}"
            ),
            collection=category,
            station_id=match.station_id,
            site=match.site,
            onc_location_code=match.station_id,
            onc_device_category=category,
            onc_data_search_url=match.extra.get("data_search_url"),
            depth_m=match.extra.get("depth"),
            summary=match.extra.get("description"),
        )
        if seconds:
            attrs["onc_resample_period_s"] = seconds
            attrs["comment"] = (
                f"Server-side resampled to {seconds} s by ONC. Pass onc_resample_seconds=0 for "
                "raw samples, which can be 1 Hz."
            )
        if citations:
            attrs["citation"] = "; ".join(
                str(c.get("citation", "")) for c in citations if c.get("citation")
            )
            dois = [str(c.get("doi")) for c in citations if c.get("doi")]
            if dois:
                attrs["doi"] = "; ".join(dois)
                attrs["references"] = "; ".join(
                    str(c.get("landingPageUrl")) for c in citations if c.get("landingPageUrl")
                ) or self.provider.terms_url
        return attrs


#: Phrases ONC uses when the answer is "nothing here", not "something went wrong".
_NO_DATA_PHRASES = (
    "not during the provided time range",
    "there is no deployment",
    "no data were found",
    "no data was found",
)


def _means_no_data(exc: UpstreamError) -> bool:
    """Is this 400 an empty result wearing an error's clothes?

    ONC answers "that instrument exists but was not deployed in your window" with HTTP 400 and
    a prose explanation, which is the same trap ERDDAP sets with its 404s. Treating it as a
    failure would make one quiet instrument in a radius abort the whole query, so it is
    translated to "no data" — while anything else, including a bad token, still raises.
    """
    if exc.status != 400:
        return False
    detail = (exc.detail or "").lower()
    return any(phrase in detail for phrase in _NO_DATA_PHRASES)


def _iso(stamp: pd.Timestamp) -> str:
    """ONC wants millisecond-precision UTC instants."""
    return stamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _safe(text: Any) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(text)) or "unknown"
