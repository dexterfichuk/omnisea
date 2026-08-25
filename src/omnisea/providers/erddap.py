"""ERDDAP — one adapter for the tens of thousands of datasets served by ERDDAP installations.

ERDDAP is the same software running at IOOS, CIOOS, NOAA CoastWatch, EMODnet and a few hundred
other institutions, and every installation answers the same handful of URLs. That is why one
adapter is the largest coverage win available: point it at a different ``erddap_server`` and the
whole catalogue of that institution becomes queryable with no new code.

Two protocols, two sources, one seam:

* ``tabledap`` serves station and platform records as tables, so
  :class:`ErddapTableSource` reuses omnisea's point path and returns
  :class:`~omnisea.providers.base.StationSeries`.
* ``griddap`` serves gridded fields over OPeNDAP, so :class:`ErddapGridSource` returns a **lazy**
  :class:`xarray.Dataset` tagged with ``omnisea_node_path``. Nothing is read until the caller
  actually indexes it.

**No field table is hardcoded here.** Every ERDDAP dataset publishes its own CF metadata —
``standard_name``, ``units``, ``cell_methods``, ``ancillary_variables`` — through
``/info/{dataset_id}/index.json``, and this adapter reads that and passes it through. Inventing a
mapping for a catalogue this size would be both impossible and wrong: the dataset author knows
what they measured. The consequence is that :attr:`ErddapSource.fields` is empty at class level
and the real table is built per dataset at fetch time.

**Why not erddapy.** ``erddapy`` builds these URLs and hands back a DataFrame, which is the easy
half. The hard half is routing through :mod:`omnisea.http` so that the retry policy, the global
concurrency cap, the User-Agent and the payload ceiling apply — and erddapy reads with its own
``pandas.read_csv`` call, outside all of that. Re-implementing the URL strings is a few lines;
re-implementing the safety is not. So this module talks to the REST endpoints directly and the
``erddap`` extra stays optional and unused.

Three upstream behaviours shape the code, all verified live:

* **Zero results are HTTP 404.** ``search/advanced.json``, ``allDatasets`` and ``tabledap`` all
  answer an empty result set with ``404 ... produced no matching results``. That is not a
  failure, so it is translated to "nothing here" — while a 404 saying ``Currently unknown
  datasetID`` still raises, because that one really is wrong.
* **``allDatasets`` is not always populated.** CIOOS Pacific publishes null bounds for every row
  of ``allDatasets`` while its search index knows the bounds perfectly well; IOOS Sensors fills
  both. Neither endpoint alone lists every dataset in a box, so both are consulted and the ids
  unioned rather than trusting whichever answers first.
* **ERDDAP publishes no row count.** There is no cheap "how many rows would this be?" call, so
  the estimate comes from ``time_coverage_resolution`` where the dataset declares it and from a
  documented assumption where it does not — and the ceiling is additionally enforced against the
  rows actually returned, so a bad estimate cannot become an unbounded download.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import pandas as pd
import xarray as xr

from .. import cf
from ..errors import PayloadTooLargeError, ProviderError, QueryError, UpstreamError
from ..http import DEFAULT_MAX_WORKERS, chunk_time, get_json, map_threads
from ..query import BBox, Query
from .base import (
    Provider,
    RetrievalSource,
    StationMatch,
    StationSeries,
    drop_orphan_qc,
    frame_from_records,
    trim_to_window,
)

log = logging.getLogger("omnisea.erddap")

__all__ = [
    "ErddapProvider",
    "ErddapTableSource",
    "ErddapGridSource",
    "DatasetInfo",
    "parse_info",
    "clear_cache",
    "DEFAULT_SERVER",
]

#: CIOOS Pacific: a Canadian regional ERDDAP, on-mission for this library and small enough that a
#: first query returns quickly. Override per query with ``erddap_server=``.
DEFAULT_SERVER = "https://data.cioospacific.ca/erddap"

#: How many datasets discovery will describe before it refuses. Each one costs an ``/info``
#: request, and a wide griddap query on a national server matches a couple of hundred — asking
#: for all of them would hammer a public service to build a catalogue nobody can read.
DEFAULT_MAX_DATASETS = 25

#: Assumed sampling rate when a dataset publishes no interval at all — ten minutes, a common
#: station reporting rate. It only ever feeds the Catalog estimate and the request chunking; the
#: row ceiling is enforced a second time against the rows actually returned, so a dataset that
#: samples faster than this cannot turn a bad guess into an unbounded download.
DEFAULT_SAMPLES_PER_DAY = 144.0

#: Rows to aim for in one tabledap request. Long windows are split to this size rather than asked
#: for in one go, because ERDDAP builds the whole response in memory before sending it.
ROWS_PER_REQUEST = 100_000

#: Columns tabledap returns that are position or identity rather than measurement.
_TIME_NAMES = frozenset({"time"})
_LATITUDE_NAMES = frozenset({"latitude", "lat"})
_LONGITUDE_NAMES = frozenset({"longitude", "lon"})

#: Per-sample position is renamed on the way out. ``series_to_dataset`` assigns the station
#: position as scalar ``latitude``/``longitude`` coordinates, which would silently overwrite
#: same-named columns — and for a glider or a ferry those columns are the most important thing
#: in the file. Renaming keeps both.
_POSITION_RENAME = {
    "latitude": "sample_latitude",
    "lat": "sample_latitude",
    "longitude": "sample_longitude",
    "lon": "sample_longitude",
}

#: Standard names ERDDAP/IOOS give to QC companion variables. QARTOD publishes one per test —
#: ``spike_test_quality_flag``, ``gross_range_test_quality_flag`` — so the suffix matters as much
#: as the exact names.
_QC_STANDARD_NAMES = frozenset({"aggregate_quality_flag", "quality_flag", "status_flag"})
_QC_SUFFIXES = ("_qc_agg", "_qc_tests", "_qartod_aggregate", "_qc", "_flag", "_flags")

#: Standard names that describe where or when a sample was taken rather than what was measured.
_COORDINATE_STANDARD_NAMES = frozenset(
    {"time", "latitude", "longitude", "depth", "altitude", "height"}
)

_info_cache: dict[tuple[str, str], DatasetInfo] = {}
_lock = threading.Lock()


def clear_cache() -> None:
    """Drop cached ``/info`` responses (used by tests)."""
    with _lock:
        _info_cache.clear()


# --------------------------------------------------------------------------- metadata model


@dataclass(frozen=True)
class DatasetInfo:
    """What ``/info/{dataset_id}/index.json`` says about one ERDDAP dataset.

    This is the whole basis of the adapter's CF description: the standard names, units and cell
    methods below are the dataset author's, not omnisea's.
    """

    dataset_id: str
    global_attrs: Mapping[str, str] = dc_field(default_factory=dict)
    #: Variable name -> its attributes, in the order ERDDAP declared them.
    variables: Mapping[str, Mapping[str, str]] = dc_field(default_factory=dict)
    #: Gridded axes, name -> ERDDAP's ``nValues=..., evenlySpaced=...`` description.
    dimensions: Mapping[str, str] = dc_field(default_factory=dict)

    # ------------------------------------------------------------------ identity

    @property
    def title(self) -> str:
        return str(self.global_attrs.get("title") or self.dataset_id)

    @property
    def institution(self) -> str:
        return str(
            self.global_attrs.get("institution")
            or self.global_attrs.get("creator_institution")
            or ""
        )

    @property
    def license(self) -> str:
        return str(self.global_attrs.get("license") or "")

    @property
    def cdm_data_type(self) -> str:
        return str(self.global_attrs.get("cdm_data_type") or "")

    @property
    def station_variable(self) -> str | None:
        """The variable holding the station id, per ``cdm_timeseries_variables``.

        ERDDAP lists the timeseries-identifying variables in declaration order, position last, so
        the first entry that is not a coordinate is the identifier.
        """
        declared = self.global_attrs.get("cdm_timeseries_variables") or ""
        for name in (n.strip() for n in str(declared).split(",")):
            if name and name in self.variables and not _is_position_or_time(name):
                return name
        return None

    # ------------------------------------------------------------------ extent

    @property
    def bounds(self) -> BBox | None:
        """Published spatial extent, or ``None`` when the dataset does not declare one."""
        west = _as_float(self.global_attrs.get("geospatial_lon_min"))
        east = _as_float(self.global_attrs.get("geospatial_lon_max"))
        south = _as_float(self.global_attrs.get("geospatial_lat_min"))
        north = _as_float(self.global_attrs.get("geospatial_lat_max"))
        if None in (west, east, south, north):
            return _bounds_from_actual_range(self.variables)
        return BBox(float(west), float(south), float(east), float(north))

    @property
    def first(self) -> pd.Timestamp | None:
        return _as_timestamp(self.global_attrs.get("time_coverage_start"))

    @property
    def last(self) -> pd.Timestamp | None:
        return _as_timestamp(self.global_attrs.get("time_coverage_end"))

    @property
    def resolution(self) -> pd.Timedelta | None:
        """Sampling interval, from whichever of the two places the dataset records it.

        Tables declare ``time_coverage_resolution`` (``PT30M00S``); grids usually do not, but
        ERDDAP measures the time axis itself and reports ``averageSpacing`` on the dimension,
        which is the better number anyway because it is observed rather than asserted.
        """
        for raw in (
            self.global_attrs.get("time_coverage_resolution"),
            _average_spacing(self.dimensions.get("time")),
        ):
            if not raw:
                continue
            try:
                value = pd.Timedelta(str(raw))
            except (ValueError, TypeError):
                continue
            if value > pd.Timedelta(0):
                return value
        return None

    @property
    def samples_per_day(self) -> float:
        resolution = self.resolution
        if resolution is None:
            return DEFAULT_SAMPLES_PER_DAY
        return float(pd.Timedelta(days=1) / resolution)

    # ------------------------------------------------------------------ variables

    @property
    def standard_names(self) -> tuple[str, ...]:
        """CF standard names this dataset advertises, excluding coordinates and QC flags.

        This is the Catalog's "what is here" column, so it lists measured quantities. Position,
        time and the QARTOD flag variables are all real and all still returned by ``fetch``;
        they just are not what someone reading a catalogue means by "what does it measure".
        """
        out: list[str] = []
        for name, attrs in self.variables.items():
            if _is_position_or_time(name):
                continue
            sn = str(attrs.get("standard_name") or "")
            if sn and not _is_coordinate_or_flag_name(sn) and sn not in out:
                out.append(sn)
        return tuple(out)

    def qc_map(self) -> dict[str, list[str]]:
        """Measurement variable -> its QC companion variables, from ``ancillary_variables``.

        CF already has a way to say "this variable holds the flags for that one", and ERDDAP
        publishes it, so the flags are found by reading the declaration rather than by guessing
        at name suffixes.
        """
        out: dict[str, list[str]] = {}
        for name, attrs in self.variables.items():
            named = str(attrs.get("ancillary_variables") or "").replace(",", " ").split()
            companions = [
                other
                for other in named
                if other in self.variables and other != name and self._is_qc_variable(other)
            ]
            if companions:
                out[name] = companions
        return out

    def _is_qc_variable(self, name: str) -> bool:
        sn = str(self.variables.get(name, {}).get("standard_name") or "")
        if sn in _QC_STANDARD_NAMES:
            return True
        return name.lower().endswith(_QC_SUFFIXES)

    def recognizes(self, wanted: frozenset[str]) -> bool:
        """Does this dataset publish any of the requested names?"""
        for name, attrs in self.variables.items():
            if name in wanted or str(attrs.get("standard_name") or "") in wanted:
                return True
        return False


def parse_info(payload: Mapping[str, Any], dataset_id: str) -> DatasetInfo:
    """Turn an ``/info/{id}/index.json`` payload into a :class:`DatasetInfo`.

    The payload is a table of ``(Row Type, Variable Name, Attribute Name, Data Type, Value)``
    rows: ``attribute``/``NC_GLOBAL`` rows are the global attributes, ``attribute``/*name* rows
    belong to a variable, and ``variable``/``dimension`` rows declare what exists.
    """
    table = payload.get("table") or {}
    columns = list(table.get("columnNames") or [])
    try:
        i_type = columns.index("Row Type")
        i_var = columns.index("Variable Name")
        i_attr = columns.index("Attribute Name")
        i_value = columns.index("Value")
    except ValueError as exc:
        raise ProviderError(
            f"ERDDAP /info for {dataset_id!r} did not have the expected columns: {columns}",
            provider="erddap",
        ) from exc

    globals_: dict[str, str] = {}
    variables: dict[str, dict[str, str]] = {}
    dimensions: dict[str, str] = {}

    for row in table.get("rows") or []:
        row_type, name, attr, value = row[i_type], row[i_var], row[i_attr], row[i_value]
        if row_type == "attribute":
            if name == "NC_GLOBAL":
                globals_[str(attr)] = value
            else:
                variables.setdefault(str(name), {})[str(attr)] = value
        elif row_type == "dimension":
            dimensions[str(name)] = str(value)
            variables.setdefault(str(name), {})
        elif row_type == "variable":
            variables.setdefault(str(name), {})

    return DatasetInfo(
        dataset_id=dataset_id,
        global_attrs=globals_,
        variables=variables,
        dimensions=dimensions,
    )


# --------------------------------------------------------------------------- provider


class ErddapProvider(Provider):
    """One ERDDAP installation.

    ``base_url`` is only the default: an ERDDAP server is interchangeable with any other, so the
    server actually queried comes from ``erddap_server=`` and is recorded on every node produced,
    together with the dataset's own licence — which on ERDDAP is per dataset, not per server.
    """

    name = "erddap"
    title = "ERDDAP"
    base_url = DEFAULT_SERVER
    license = "Per-dataset; see each dataset's 'license' global attribute"
    terms_url = "https://coastwatch.pfeg.noaa.gov/erddap/information.html"

    def build_sources(self) -> Sequence[RetrievalSource]:
        return [ErddapTableSource(self), ErddapGridSource(self)]


# --------------------------------------------------------------------------- shared source


class ErddapSource(RetrievalSource):
    """Behaviour common to the ``tabledap`` and ``griddap`` halves of an ERDDAP server."""

    #: ``tabledap`` or ``griddap`` — the ERDDAP protocol this source speaks.
    protocol: str = ""
    #: ``allDatasets.dataStructure`` value that corresponds to :attr:`protocol`.
    data_structure: str = ""

    # No class-level table: an ERDDAP dataset describes its own variables, and this adapter
    # reads that description rather than replacing it. See the module docstring.
    fields: dict[str, cf.FieldSpec] = {}

    # ------------------------------------------------------------------ configuration

    def server(self, query: Query) -> str:
        server = str(query.option("erddap_server") or self.provider.base_url).rstrip("/")
        if not server.startswith(("http://", "https://")):
            raise QueryError(
                f"erddap_server must be a full http(s) URL ending at the ERDDAP root, "
                f"e.g. {DEFAULT_SERVER!r}; got {server!r}"
            )
        return server

    def wants_anything(self, query: Query) -> bool:
        """Always in play until the datasets themselves have been read.

        The base implementation opts a source out when it does not recognise any requested name,
        which relies on a curated field table. This source has none by design — an ERDDAP server
        publishes whatever its institution measures — so opting out early would hide most of
        ERDDAP. The real filtering happens in :meth:`discover`, against each dataset's own
        declared variables.
        """
        return True

    # ------------------------------------------------------------------ discovery

    def discover(self, query: Query) -> list[StationMatch]:
        server = self.server(query)
        named = _as_list(query.option("erddap_datasets"))
        candidates = named or self._candidate_ids(query, server)
        if not candidates:
            return []

        cap = int(query.option("erddap_max_datasets", DEFAULT_MAX_DATASETS))
        if len(candidates) > cap:
            raise PayloadTooLargeError(
                f"{len(candidates)} {self.protocol} datasets on {server} overlap this query, "
                f"over the {cap} dataset ceiling. Describing each one costs a metadata request, "
                "so narrow the bbox/time window, name the datasets you want with "
                "erddap_datasets=[...], or raise erddap_max_datasets.",
                estimate=len(candidates),
                limit=cap,
            )

        infos = map_threads(
            lambda ds: self._info(server, ds),
            candidates,
            max_workers=int(query.option("max_workers", DEFAULT_MAX_WORKERS)),
            label=f"{self.name} dataset info",
        )

        wanted = cf.resolve_names(query.variables)
        matches: list[StationMatch] = []
        for info in infos:
            match = self._match_for(query, server, info, wanted, explicit=bool(named))
            if match is not None:
                matches.append(match.attach_site(query))
        log.debug("%s discovered %d dataset(s) on %s", self.name, len(matches), server)
        return matches

    def _match_for(
        self,
        query: Query,
        server: str,
        info: DatasetInfo,
        wanted: frozenset[str] | None,
        *,
        explicit: bool = False,
    ) -> StationMatch | None:
        bounds = info.bounds
        if not explicit:
            if bounds is None or not _overlaps_query(query, bounds):
                return None
            if not query.overlaps(info.first, info.last):
                return None
            if wanted is not None and not info.recognizes(wanted):
                return None

        # A dataset the caller named by hand is fetched whatever its metadata says. Plenty of
        # real ERDDAP datasets publish no geospatial extent at all — six of the nine on CIOOS
        # Pacific do not — and silently returning nothing for a dataset someone asked for by
        # name, because its publisher skipped an attribute, would be the wrong kind of strict.
        # Where the extent is genuinely unknown the position is NaN rather than invented; fetch
        # fills it in from the rows.
        lat, lon = bounds.centre if bounds is not None else (float("nan"), float("nan"))
        return self.new_match(
            station_id=info.dataset_id,
            name=info.title,
            lat=float(lat),
            lon=float(lon),
            variables=info.standard_names,
            n_rows_est=self._estimate_rows(query, info),
            first=info.first,
            last=info.last,
            extra={
                "server": server,
                "dataset_id": info.dataset_id,
                "protocol": self.protocol,
                "bounds": tuple(bounds) if bounds is not None else None,
            },
        )

    def _estimate_rows(self, query: Query, info: DatasetInfo) -> int:
        """Rows the window would pull, over the part of it the dataset actually covers."""
        start = max(query.start, info.first) if info.first is not None else query.start
        end = min(query.end, info.last) if info.last is not None else query.end
        days = max((end - start) / pd.Timedelta(days=1), 0.0)
        return int(days * info.samples_per_day)

    # ------------------------------------------------------------------ candidate listing

    def _candidate_ids(self, query: Query, server: str) -> list[str]:
        """Dataset ids that might answer this query, cheaply and without duplicates.

        ``allDatasets`` and the search index are both consulted: they disagree in practice (see
        the module docstring), and taking their union is the only way a dataset one of them
        forgot still shows up.
        """
        found: list[str] = []
        for dataset_id in self._from_all_datasets(query, server) + self._from_search(query, server):
            # allDatasets is ERDDAP's own catalogue table, and it lists itself.
            if dataset_id not in found and dataset_id != "allDatasets":
                found.append(dataset_id)
        return found

    def _from_all_datasets(self, query: Query, server: str) -> list[str]:
        """Ids from ``tabledap/allDatasets``, filtered server-side by structure, box and time."""
        constraints = [
            "datasetID,dataStructure,minLongitude,maxLongitude,minLatitude,maxLatitude"
            ",minTime,maxTime",
            f'dataStructure="{self.data_structure}"',
        ]
        bbox = query.bbox
        if bbox is not None:
            constraints += [
                f"maxLongitude>={bbox.west}",
                f"minLongitude<={bbox.east}",
                f"maxLatitude>={bbox.south}",
                f"minLatitude<={bbox.north}",
            ]
        constraints += [
            f"maxTime>={query.start.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            f"minTime<={query.end.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        ]
        url = f"{server}/tabledap/allDatasets.json?" + "&".join(constraints)
        payload = self._get(url, None)
        if payload is None:
            return []
        rows = _table_rows(payload)
        return [str(row["datasetID"]) for row in rows if row.get("datasetID")]

    def _from_search(self, query: Query, server: str) -> list[str]:
        """Ids from ``search/advanced.json``, ERDDAP's own spatial/temporal index."""
        params: dict[str, Any] = {
            "page": 1,
            "itemsPerPage": 1000,
            "protocol": self.protocol,
            "minTime": query.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "maxTime": query.end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if query.bbox is not None:
            params.update(
                minLon=query.bbox.west,
                maxLon=query.bbox.east,
                minLat=query.bbox.south,
                maxLat=query.bbox.north,
            )
        search_for = query.option("erddap_search")
        if search_for:
            params["searchFor"] = str(search_for)

        payload = self._get(f"{server}/search/advanced.json", params)
        if payload is None:
            return []
        return [str(row["Dataset ID"]) for row in _table_rows(payload) if row.get("Dataset ID")]

    # ------------------------------------------------------------------ http

    def _get(self, url: str, params: Mapping[str, Any] | None) -> Any | None:
        """GET JSON, mapping ERDDAP's "no matching results" 404 onto ``None``.

        ERDDAP answers an empty result set with a 404 rather than an empty table, so treating
        every 404 as a failure would turn "no data here" into an error on a routine query. A 404
        that names an unknown datasetID is a different thing and still raises.
        """
        try:
            return get_json(url, dict(params or {}), provider=self.name)
        except UpstreamError as exc:
            if exc.status == 404 and "produced no matching results" in (exc.detail or ""):
                log.debug("erddap: no matching results for %s", url)
                return None
            raise

    def _info(self, server: str, dataset_id: str) -> DatasetInfo:
        """Read (and memoize) one dataset's metadata."""
        key = (server, dataset_id)
        with _lock:
            cached = _info_cache.get(key)
        if cached is not None:
            return cached
        payload = self._get(f"{server}/info/{dataset_id}/index.json", None)
        if payload is None:
            raise ProviderError(
                f"ERDDAP server {server} has no metadata for dataset {dataset_id!r}",
                provider=self.name,
            )
        info = parse_info(payload, dataset_id)
        with _lock:
            _info_cache[key] = info
        return info

    # ------------------------------------------------------------------ node attributes

    def _node_attrs(self, info: DatasetInfo, server: str, **extra: Any) -> dict[str, Any]:
        """Node attributes, with the dataset's own licence and institution taking precedence.

        On ERDDAP the licence belongs to the dataset, not the server: one installation hosts data
        from a dozen institutions under a dozen terms. Stamping the provider's placeholder over
        the dataset's real licence would misattribute it.
        """
        return self.base_attrs(
            title=info.title,
            summary=str(info.global_attrs.get("summary") or "") or None,
            institution=info.institution or None,
            license=info.license or None,
            erddap_server=server,
            erddap_dataset_id=info.dataset_id,
            erddap_protocol=self.protocol,
            source_url=f"{server}/{self.protocol}/{info.dataset_id}",
            cdm_data_type=info.cdm_data_type or None,
            **extra,
        )


# --------------------------------------------------------------------------- tabledap


class ErddapTableSource(ErddapSource):
    """``tabledap`` — station, mooring, profile and trajectory records, as point time series."""

    name = "erddap_tabledap"
    title = "ERDDAP tabledap"
    node_path = "in_situ/erddap"
    feature_type = "timeSeries"
    protocol = "tabledap"
    data_structure = "table"

    def fetch(self, query: Query, matches: list[StationMatch]) -> list[StationSeries]:
        results = map_threads(
            lambda m: self._fetch_dataset(query, m),
            matches,
            max_workers=int(query.option("max_workers", DEFAULT_MAX_WORKERS)),
            label=f"{self.name} dataset",
        )
        return [series for group in results for series in group]

    def _fetch_dataset(self, query: Query, match: StationMatch) -> list[StationSeries]:
        server = match.require("server")
        dataset_id = match.require("dataset_id")
        info = self._info(server, dataset_id)

        estimate = self._estimate_rows(query, info)
        if estimate > query.max_rows:
            raise PayloadTooLargeError(
                f"{dataset_id} on {server} would return about {estimate:,} rows, over the "
                f"{query.max_rows:,} row ceiling. Narrow the time window or raise max_rows.",
                estimate=estimate,
                limit=query.max_rows,
            )

        rows = self._download(query, server, info)
        if not rows:
            return []
        return self.series_from_rows(query, match, info, rows)

    def _download(
        self, query: Query, server: str, info: DatasetInfo
    ) -> list[dict[str, Any]]:
        """Pull the window, split into requests ERDDAP can build without running out of memory."""
        max_days = max(ROWS_PER_REQUEST / max(info.samples_per_day, 1e-6), 1.0)
        url = f"{server}/tabledap/{info.dataset_id}.json"
        space = _space_constraints(info, query)

        rows: list[dict[str, Any]] = []
        for start, end in chunk_time(query.start, query.end, max_days=max_days):
            # An empty variable list means "every variable"; the constraints follow it. The query
            # string is passed whole because ERDDAP's constraint syntax is positional, not a set
            # of named parameters.
            constraint = (
                f"&time>={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                f"&time<={end.strftime('%Y-%m-%dT%H:%M:%SZ')}" + space
            )
            payload = self._get(f"{url}?{constraint}", None)
            if payload is None:
                continue
            rows.extend(_table_rows(payload))
            if len(rows) > query.max_rows:
                raise PayloadTooLargeError(
                    f"{info.dataset_id} on {server} returned more than the "
                    f"{query.max_rows:,} row ceiling. Narrow the time window or raise max_rows.",
                    estimate=len(rows),
                    limit=query.max_rows,
                )
        return rows

    # ------------------------------------------------------------------ shaping

    def series_from_rows(
        self,
        query: Query,
        match: StationMatch,
        info: DatasetInfo,
        rows: list[Mapping[str, Any]],
    ) -> list[StationSeries]:
        """Turn tabledap rows into CF-described series — one per station in the response.

        A tabledap dataset is not always one station: a cruise or a sensor network ships every
        platform in one table. Collapsing those onto a single time index would silently discard
        every station but one, because two platforms report at the same instant, so the rows are
        split on the dataset's own ``cdm_timeseries_variables`` identifier first.
        """
        station_var = info.station_variable
        groups = _split_by_station(rows, station_var)
        table = _field_table(info, present=_ordered_keys(rows))
        skip = _skip_columns(info, station_var)
        primary_qc = _primary_qc_names(info)
        to_cf = self.to_cf_units(query)

        out: list[StationSeries] = []
        for station_id, group in groups.items():
            specs = cf.resolve_fields(
                table,
                _ordered_keys(group),
                include_unmapped=self.include_unmapped(query),
                skip=skip,
                is_qc=lambda raw: raw in primary_qc,
                units_for=lambda raw: _units_of(info, raw),
            )
            frame, var_attrs = self._frame_for(query, group, specs, to_cf)
            if frame.empty:
                continue

            member = self._member_match(match, info, station_id, group)
            path = f"{self.node_path}/{_safe(info.dataset_id)}"
            if station_id is not None and len(groups) > 1:
                path = f"{path}/{_safe(station_id)}"
            out.append(
                StationSeries(
                    match=member,
                    frame=frame,
                    node_path=path,
                    attrs=self._node_attrs(
                        info,
                        match.require("server"),
                        station_id=member.station_id,
                        site=member.site,
                        time_coverage_resolution=str(
                            info.global_attrs.get("time_coverage_resolution") or ""
                        )
                        or None,
                    ),
                    var_attrs=var_attrs,
                )
            )
        return out

    def _frame_for(
        self,
        query: Query,
        rows: list[Mapping[str, Any]],
        specs: Mapping[str, cf.FieldSpec],
        to_cf: bool,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        for row in rows:
            time_value = row.get("time")
            if time_value is None:
                continue
            record: dict[str, Any] = {"time": time_value}
            for raw, spec in specs.items():
                record[spec.var] = cf.convert(row.get(raw), spec, to_cf_units=to_cf)
                if spec.qc_field and row.get(spec.qc_field) is not None:
                    record[f"{spec.var}_qc"] = row.get(spec.qc_field)
            records.append(record)

        frame = drop_orphan_qc(
            trim_to_window(frame_from_records(records), query.start, query.end)
        )
        var_attrs: dict[str, dict[str, Any]] = {}
        if frame is None or frame.empty:
            return pd.DataFrame(), var_attrs
        for spec in specs.values():
            if spec.var in frame.columns:
                var_attrs[spec.var] = cf.cf_attrs(spec, to_cf_units=to_cf)
            qc_col = f"{spec.var}_qc"
            if qc_col in frame.columns:
                var_attrs[qc_col] = {
                    "long_name": f"quality flag for {spec.var}",
                    "source_field": spec.qc_field or "",
                    cf.MAPPED_ATTR: 0,
                }
        return frame, var_attrs

    def _member_match(
        self,
        match: StationMatch,
        info: DatasetInfo,
        station_id: Any,
        rows: list[Mapping[str, Any]],
    ) -> StationMatch:
        """The match this series belongs to: the dataset's, or a per-station copy of it."""
        lat, lon = _mean_position(rows)
        if station_id is None:
            if lat is None or lon is None:
                return match
            # A single-station dataset positions better from its rows than from its declared
            # bounding box, which for a point station are the same thing anyway.
            member = self.new_match(
                station_id=match.station_id,
                name=match.name,
                lat=lat,
                lon=lon,
                variables=match.variables,
                n_rows_est=match.n_rows_est,
                first=match.first,
                last=match.last,
                extra=dict(match.extra),
            )
            member.site, member.distance_km = match.site, match.distance_km
            return member

        member = self.new_match(
            station_id=f"{info.dataset_id}:{station_id}",
            name=f"{info.title} — {station_id}",
            lat=lat if lat is not None else match.lat,
            lon=lon if lon is not None else match.lon,
            variables=match.variables,
            n_rows_est=len(rows),
            first=match.first,
            last=match.last,
            extra={**match.extra, "erddap_station": station_id},
        )
        member.site, member.distance_km = match.site, match.distance_km
        return member


# --------------------------------------------------------------------------- griddap


class ErddapGridSource(ErddapSource):
    """``griddap`` — gridded fields, returned lazily over OPeNDAP.

    The subset is expressed as an :meth:`xarray.Dataset.sel` on the open remote dataset, so no
    bytes move until the caller indexes the result. That is the point of the gridded path: a user
    can put a decade of a global SST analysis in their tree and only pay for the pixels they read.
    """

    name = "erddap_griddap"
    title = "ERDDAP griddap"
    node_path = "gridded/erddap"
    feature_type = "grid"
    protocol = "griddap"
    data_structure = "grid"

    def fetch(self, query: Query, matches: list[StationMatch]) -> list[xr.Dataset]:
        # Deliberately serial: each open is an OPeNDAP handshake against a public server, and the
        # cost is a couple of metadata round-trips, not a download worth parallelising.
        out: list[xr.Dataset] = []
        for match in matches:
            dataset = self._open_subset(query, match)
            if dataset is not None:
                out.append(dataset)
        return out

    def _open_subset(self, query: Query, match: StationMatch) -> xr.Dataset | None:
        server = match.require("server")
        dataset_id = match.require("dataset_id")
        info = self._info(server, dataset_id)

        remote = self._open(f"{server}/griddap/{dataset_id}")
        selection = _grid_selection(remote, query)
        subset = remote.sel(selection) if selection else remote
        if any(size == 0 for size in subset.sizes.values()):
            log.debug("griddap %s does not intersect the query after subsetting", dataset_id)
            remote.close()
            return None

        for name, variable in subset.data_vars.items():
            variable.attrs.setdefault(
                cf.MAPPED_ATTR, 1 if variable.attrs.get("standard_name") else 0
            )
            variable.attrs.setdefault("source_field", str(name))

        cells = int(max((v.size for v in subset.data_vars.values()), default=0))
        subset.attrs = {
            **subset.attrs,
            **self._node_attrs(
                info,
                server,
                omnisea_node_path=f"{self.node_path}/{_safe(dataset_id)}",
                omnisea_cells_estimate=cells,
                site=match.site,
            ),
        }
        return subset

    @staticmethod
    def _open(url: str) -> xr.Dataset:
        """Open an ERDDAP griddap endpoint as a lazy dataset.

        This is the one request in the module that does not go through :mod:`omnisea.http`: DAP
        is a binary protocol spoken by the netCDF library, and routing it through a JSON session
        would mean downloading the array instead of referencing it.
        """
        if not (
            importlib.util.find_spec("netCDF4") is not None
            or importlib.util.find_spec("pydap") is not None
        ):
            raise ProviderError(
                "reading ERDDAP griddap needs an OPeNDAP-capable engine; install one with "
                'pip install "omnisea[netcdf]"',
                provider="erddap_griddap",
            )
        # Dask keeps a big grid chunked rather than one array-shaped lazy read, but it is not a
        # hard dependency; without it xarray's own lazy indexing still defers every byte.
        chunks: dict[str, Any] | None = {} if importlib.util.find_spec("dask") else None
        try:
            return xr.open_dataset(url, chunks=chunks, decode_timedelta=False)
        except Exception as exc:  # noqa: BLE001 - surfaced as an omnisea error
            raise UpstreamError(
                f"could not open griddap dataset: {exc}", provider="erddap_griddap", url=url
            ) from exc


# --------------------------------------------------------------------------- field tables


def _field_table(info: DatasetInfo, *, present: Iterable[str]) -> dict[str, cf.FieldSpec]:
    """Build the CF field table for one dataset from its own published metadata.

    The output variable takes the CF standard name where the dataset gives one and that name is
    unambiguous within the dataset; where two variables share a standard name — the same quantity
    at two depths, say — both keep their ERDDAP names, because a ``_2`` suffix would tell a
    reader nothing about which is which.
    """
    present = list(present)
    qc_map = info.qc_map()
    primary_qc = _primary_qc_names(info)
    all_qc = {name for names in qc_map.values() for name in names}

    emitted = [
        name
        for name in present
        if name in info.variables and name not in all_qc and not _is_time(name)
    ]
    counts: dict[str, int] = {}
    for name in emitted:
        sn = str(info.variables[name].get("standard_name") or "")
        if sn:
            counts[sn] = counts.get(sn, 0) + 1

    table: dict[str, cf.FieldSpec] = {}
    for name in emitted:
        attrs = info.variables[name]
        standard_name = str(attrs.get("standard_name") or "")
        var = standard_name if (standard_name and counts.get(standard_name, 0) == 1) else name
        if name in _POSITION_RENAME:
            var = _POSITION_RENAME[name]
        qc_field = next((q for q in qc_map.get(name, ()) if q in primary_qc), None)

        extra: dict[str, Any] = {"source_field": name}
        if name in _POSITION_RENAME:
            extra["comment"] = (
                "Per-sample position as published by the dataset; the scalar latitude/longitude "
                "coordinates hold the station position omnisea matched on."
            )
        table[name] = cf.FieldSpec(
            var=var,
            standard_name=standard_name,
            units=_units_of(info, name),
            cell_methods=str(attrs.get("cell_methods") or "") or None,
            long_name=str(attrs.get("long_name") or "") or None,
            qc_field=qc_field,
            extra_attrs=extra,
        )
    return table


def _primary_qc_names(info: DatasetInfo) -> frozenset[str]:
    """The one QC variable per measurement that becomes ``<var>_qc``.

    IOOS datasets publish two: an aggregate QARTOD flag and a bitmask of the individual tests.
    omnisea carries one flag column per variable, so the aggregate is promoted and any others
    still travel under their own names rather than being dropped.
    """
    chosen: set[str] = set()
    for _parent, companions in info.qc_map().items():
        aggregate = next(
            (
                name
                for name in companions
                if str(info.variables[name].get("standard_name") or "")
                == "aggregate_quality_flag"
            ),
            None,
        )
        chosen.add(aggregate or companions[0])
    return frozenset(chosen)


def _skip_columns(info: DatasetInfo, station_var: str | None) -> frozenset[str]:
    """Columns that identify or time-stamp a row rather than measure something."""
    skip = set(_TIME_NAMES)
    if station_var:
        # Station identity becomes the node's station_id coordinate, not a column.
        skip.add(station_var)
    return frozenset(skip)


def _units_of(info: DatasetInfo, name: str) -> str | None:
    units = info.variables.get(name, {}).get("units")
    return str(units) if units not in (None, "") else None


def _space_constraints(info: DatasetInfo, query: Query) -> str:
    """tabledap constraints clipping a request to the requested area and depth range.

    This is not an optimization, it is the difference between a query and a bulk download. NOAA's
    ``cwwcNDBCMet`` is one dataset holding every NDBC buoy on Earth, so a request for a box off
    Vancouver Island that carried no spatial constraint would return 700 stations. The
    consequence worth knowing is that a moving platform comes back **clipped to the area you
    asked for** rather than as its whole deployment.
    """
    parts: list[str] = []
    if query.bbox is not None:
        lat_name = _named_variable(info, _LATITUDE_NAMES)
        lon_name = _named_variable(info, _LONGITUDE_NAMES)
        if lat_name and lon_name:
            parts += [
                f"&{lat_name}>={query.bbox.south}",
                f"&{lat_name}<={query.bbox.north}",
                f"&{lon_name}>={query.bbox.west}",
                f"&{lon_name}<={query.bbox.east}",
            ]
    if query.depth is not None and "depth" in info.variables:
        # Only a variable actually called `depth`: `z` is positive-up on some datasets and
        # positive-down on others, and guessing the sign would silently invert the range.
        parts += [
            f"&depth>={min(query.depth)}",
            f"&depth<={max(query.depth)}",
        ]
    return "".join(parts)


def _named_variable(info: DatasetInfo, names: frozenset[str]) -> str | None:
    for name in names:
        if name in info.variables:
            return name
    return None


# --------------------------------------------------------------------------- row helpers


def _table_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """ERDDAP's ``{"table": {"columnNames": [...], "rows": [[...]]}}`` as a list of dicts."""
    table = payload.get("table") or {}
    columns = [str(c) for c in table.get("columnNames") or []]
    return [dict(zip(columns, row, strict=False)) for row in table.get("rows") or []]


def _ordered_keys(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    return list(seen)


def _split_by_station(
    rows: list[Mapping[str, Any]], station_var: str | None
) -> dict[Any, list[Mapping[str, Any]]]:
    """Group rows by station, or return the whole table under ``None`` when there is one."""
    if not station_var:
        return {None: rows}
    values = {row.get(station_var) for row in rows}
    values.discard(None)
    if len(values) <= 1:
        return {None: rows}
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get(station_var), []).append(row)
    return groups


def _mean_position(rows: Iterable[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    """Mean of the position columns, for datasets that report it per row."""
    lats = [_as_float(_first_of(row, _LATITUDE_NAMES)) for row in rows]
    lons = [_as_float(_first_of(row, _LONGITUDE_NAMES)) for row in rows]
    lats = [v for v in lats if v is not None]
    lons = [v for v in lons if v is not None]
    if not lats or not lons:
        return None, None
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _first_of(row: Mapping[str, Any], names: frozenset[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


# --------------------------------------------------------------------------- grid subsetting


def _grid_selection(dataset: xr.Dataset, query: Query) -> dict[str, slice]:
    """The lazy ``sel`` that clips a griddap dataset to the query, in each axis's own order.

    Grids disagree about direction and about longitude convention — latitude descends as often as
    it ascends, and a global grid may run 0-360 — so each slice is built from the coordinate's
    own values rather than from the query's orientation.
    """
    selection: dict[str, slice] = {}

    time_name = _axis_name(dataset, "T", _TIME_NAMES)
    if time_name is not None:
        # griddap times decode to naive datetime64; omnisea's are tz-aware UTC.
        selection[time_name] = _ordered_slice(
            dataset[time_name], query.start.tz_localize(None), query.end.tz_localize(None)
        )

    if query.bbox is not None:
        lat_name = _axis_name(dataset, "Y", _LATITUDE_NAMES)
        if lat_name is not None:
            selection[lat_name] = _ordered_slice(
                dataset[lat_name], query.bbox.south, query.bbox.north
            )
        lon_name = _axis_name(dataset, "X", _LONGITUDE_NAMES)
        if lon_name is not None:
            west, east = _match_longitude_convention(dataset[lon_name], query.bbox)
            selection[lon_name] = _ordered_slice(dataset[lon_name], west, east)

    if query.depth is not None:
        depth_name = _axis_name(dataset, "Z", frozenset({"depth", "altitude", "z"}))
        if depth_name is not None:
            selection[depth_name] = _ordered_slice(
                dataset[depth_name], min(query.depth), max(query.depth)
            )
    return selection


def _axis_name(dataset: xr.Dataset, axis: str, names: frozenset[str]) -> str | None:
    for name, coord in dataset.coords.items():
        if str(coord.attrs.get("axis") or "").upper() == axis and name in dataset.dims:
            return str(name)
    for name in names:
        if name in dataset.dims and name in dataset.coords:
            return name
    return None


def _ordered_slice(coord: xr.DataArray, low: Any, high: Any) -> slice:
    """``slice(low, high)``, reversed when the coordinate runs the other way."""
    if coord.size >= 2 and coord.values[0] > coord.values[-1]:
        return slice(high, low)
    return slice(low, high)


def _match_longitude_convention(coord: xr.DataArray, bbox: BBox) -> tuple[float, float]:
    """Express the query's longitudes the way this grid does (-180..180 or 0..360)."""
    if coord.size == 0:
        return bbox.west, bbox.east
    values = coord.values
    if float(values.max()) > 180.0 and bbox.west < 0:
        return bbox.west + 360.0, bbox.east + 360.0
    return bbox.west, bbox.east


# --------------------------------------------------------------------------- misc helpers


def _overlaps_query(query: Query, bounds: BBox) -> bool:
    """Does a dataset's published extent intersect what was asked for?

    A single-point extent is a fixed station, so the site radius decides. Anything with real
    extent is a cruise, glider or grid, and the honest question is whether it passed through the
    area at all — which is a box intersection, not a distance from its centre.
    """
    if _is_point(bounds):
        return query.contains(bounds.south, bounds.west)
    boxes = [site.bbox for site in query.sites] or ([query.bbox] if query.bbox else [])
    if not boxes:
        return True
    return any(_boxes_intersect(bounds, box) for box in boxes)


def _is_point(bounds: BBox) -> bool:
    return math.isclose(bounds.south, bounds.north, abs_tol=1e-6) and math.isclose(
        bounds.west, bounds.east, abs_tol=1e-6
    )


def _boxes_intersect(a: BBox, b: BBox) -> bool:
    return not (a.east < b.west or a.west > b.east or a.north < b.south or a.south > b.north)


def _bounds_from_actual_range(variables: Mapping[str, Mapping[str, str]]) -> BBox | None:
    """Fall back to the latitude/longitude ``actual_range`` when no geospatial globals exist."""
    lat = _range_of(variables, _LATITUDE_NAMES)
    lon = _range_of(variables, _LONGITUDE_NAMES)
    if lat is None or lon is None:
        return None
    return BBox(lon[0], lat[0], lon[1], lat[1])


def _range_of(
    variables: Mapping[str, Mapping[str, str]], names: frozenset[str]
) -> tuple[float, float] | None:
    for name in names:
        raw = variables.get(name, {}).get("actual_range")
        if not raw:
            continue
        parts = [_as_float(p) for p in str(raw).split(",")]
        if len(parts) == 2 and None not in parts:
            return min(parts), max(parts)  # type: ignore[type-var]
    return None


def _average_spacing(dimension: str | None) -> str | None:
    """The ``averageSpacing=...`` tail of an ERDDAP dimension description."""
    if not dimension or "averageSpacing=" not in dimension:
        return None
    return dimension.split("averageSpacing=", 1)[1].strip() or None


def _is_coordinate_or_flag_name(standard_name: str) -> bool:
    return (
        standard_name in _COORDINATE_STANDARD_NAMES
        or standard_name in _QC_STANDARD_NAMES
        or standard_name.endswith("_quality_flag")
    )


def _is_time(name: str) -> bool:
    return name in _TIME_NAMES


def _is_position_or_time(name: str) -> bool:
    return name in _TIME_NAMES or name in _LATITUDE_NAMES or name in _LONGITUDE_NAMES


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:  # noqa: BLE001 - a malformed coverage date is not fatal
        return None
    if pd.isna(ts):
        return None
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _as_list(value: Any) -> list[str]:
    if value in (None, "", ()):
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _safe(text: Any) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(text)) or "unknown"
