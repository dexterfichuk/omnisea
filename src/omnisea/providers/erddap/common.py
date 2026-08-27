"""Behaviour the ``tabledap`` and ``griddap`` halves of an ERDDAP server share.

Discovery is the bulk of it: both protocols find candidate datasets the same way (the
``allDatasets`` table and the search index, unioned), read the same ``/info`` metadata, apply
the same extent and time filters, and answer "nothing matched" the same odd way (HTTP 404).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from ... import cf
from ...errors import PayloadTooLargeError, ProviderError, QueryError, UpstreamError
from ...http import DEFAULT_MAX_WORKERS, get_json, map_threads
from ...query import BBox, Query, register_option
from ..base import RetrievalSource, StationMatch
from .info import DatasetInfo, cached_info, parse_info, store_info

log = logging.getLogger("omnisea.erddap")

__all__ = [
    "DEFAULT_SERVER",
    "DEFAULT_MAX_DATASETS",
    "ErddapSource",
    "table_rows",
    "safe_name",
]

register_option("erddap_server", "erddap: ERDDAP server root URL (default IOOS Sensors)")
register_option("erddap_datasets", "erddap: dataset id(s) to use instead of searching the server")
register_option("erddap_search", "erddap: free-text searchFor passed to the ERDDAP search index")
register_option(
    "erddap_max_datasets", "erddap: ceiling on datasets discovery will describe (default 25)"
)

#: Chosen because it answers. CIOOS Pacific is the natural regional fit but publishes nine
#: datasets, of which one declares a geospatial extent covering seven weeks of 2021 — a user
#: trying ERDDAP against it would reasonably conclude the adapter was broken. IOOS Sensors
#: indexes thousands, including the Bamfield-area hydrometric gauges and NDBC buoys, and
#: answered a Barkley Sound query in 0.9 s against CIOOS Pacific's 4.1 s for nothing.
#: Override per query with ``erddap_server=``.
DEFAULT_SERVER = "https://erddap.sensors.ioos.us/erddap"

#: How many datasets discovery will describe before it refuses. Each one costs an ``/info``
#: request, and a wide griddap query on a national server matches a couple of hundred — asking
#: for all of them would hammer a public service to build a catalogue nobody can read.
DEFAULT_MAX_DATASETS = 25


class ErddapSource(RetrievalSource):
    """Behaviour common to the ``tabledap`` and ``griddap`` halves of an ERDDAP server."""

    #: ``tabledap`` or ``griddap`` — the ERDDAP protocol this source speaks.
    protocol: str = ""
    #: ``allDatasets.dataStructure`` value that corresponds to :attr:`protocol`.
    data_structure: str = ""

    # No class-level table: an ERDDAP dataset describes its own variables, and this adapter
    # reads that description rather than replacing it. See the package docstring.
    fields: dict[str, cf.FieldSpec] = {}
    fields_from_metadata = True

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
            unusable = self.unusable_reason(info)
            if unusable:
                if named:
                    # They asked for this one by name, so silence would look like "no data
                    # here". Name the dataset and the reason instead.
                    raise ProviderError(
                        f"{info.dataset_id} on {server} cannot be read as a time series: "
                        f"{unusable}",
                        provider=self.name,
                    )
                log.debug("%s: skipping %s — %s", self.name, info.dataset_id, unusable)
                continue
            match = self._match_for(query, server, info, wanted, explicit=bool(named))
            if match is not None:
                # _match_for already attached the site and, for datasets with real extent,
                # corrected the distance to the nearest edge. Re-attaching would undo that.
                matches.append(match)
        log.debug("%s discovered %d dataset(s) on %s", self.name, len(matches), server)
        return matches

    def unusable_reason(self, info: DatasetInfo) -> str | None:
        """Why this adapter cannot read ``info``, or ``None`` when it can.

        Checked before a match is built, so a dataset omnisea cannot serve is refused with an
        explanation rather than becoming a request the server rejects for reasons the user
        cannot act on. Subclasses that override this should call ``super()`` first.
        """
        return self._wrong_protocol_reason(info)

    def _wrong_protocol_reason(self, info: DatasetInfo) -> str | None:
        """Refuse a dataset that the *other* ERDDAP protocol serves, and name that one.

        ERDDAP splits its catalogue in two and a dataset id looks identical either way, so
        pasting one off a catalogue page into the wrong source is the easy mistake to make.
        Left to fail when the data is read it surfaces as a raw HDF5 stack trace out of the
        netCDF library (griddap asked to open a table) or a bare 404 (tabledap asked for a
        grid) — neither of which names the problem or the source that would have worked.

        Grids declare dimensions in their ``/info`` response and tables never do, so this is
        the server's own answer rather than a guess from the id or the ``cdm_data_type``.
        """
        gridded = bool(info.dimensions)
        if gridded == (self.data_structure == "grid"):
            return None
        serves = "erddap_griddap" if gridded else "erddap_tabledap"
        return (
            f"this is {'gridded' if gridded else 'tabular'} data, which {serves} reads — "
            f"{self.name} reads the other kind. Use providers=['{serves}']"
        )

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
        match = self.new_match(
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
        match.attach_site(query)
        if bounds is not None and not _is_point(bounds):
            # An extent is a track or a grid: how near it *comes*, not where its middle is.
            match.distance_km = _distance_to_extent(query, bounds)
        return match

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
        the package docstring), and taking their union is the only way a dataset one of them
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
        rows = table_rows(payload)
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
        return [str(row["Dataset ID"]) for row in table_rows(payload) if row.get("Dataset ID")]

    # ------------------------------------------------------------------ http

    def _get(self, url: str, params: dict[str, Any] | None) -> Any | None:
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
        cached = cached_info(server, dataset_id)
        if cached is not None:
            return cached
        payload = self._get(f"{server}/info/{dataset_id}/index.json", None)
        if payload is None:
            raise ProviderError(
                f"ERDDAP server {server} has no metadata for dataset {dataset_id!r}",
                provider=self.name,
            )
        info = parse_info(payload, dataset_id)
        store_info(server, dataset_id, info)
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


# --------------------------------------------------------------------------- geometry


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


def _distance_to_extent(query: Query, bounds: BBox) -> float | None:
    """Distance from the query's nearest site to the closest point of a dataset's extent.

    Zero when the extent covers the site. Measuring to the centre instead makes a glider track
    that runs past your door look 250 km away, and ``nearest=`` and ``max_distance_km=`` then
    act on that number.
    """
    if not query.sites:
        return None
    best: float | None = None
    for site in query.sites:
        lat = min(max(site.lat, bounds.south), bounds.north)
        lon = min(max(site.lon, bounds.west), bounds.east)
        distance = site.distance_km(lat, lon)
        if best is None or distance < best:
            best = distance
    return best


def _is_point(bounds: BBox) -> bool:
    return math.isclose(bounds.south, bounds.north, abs_tol=1e-6) and math.isclose(
        bounds.west, bounds.east, abs_tol=1e-6
    )


def _boxes_intersect(a: BBox, b: BBox) -> bool:
    return not (a.east < b.west or a.west > b.east or a.north < b.south or a.south > b.north)


# --------------------------------------------------------------------------- payload helpers


def table_rows(payload: Any) -> list[dict[str, Any]]:
    """ERDDAP's ``{"table": {"columnNames": [...], "rows": [[...]]}}`` as a list of dicts."""
    table = (payload or {}).get("table") or {}
    columns = [str(c) for c in table.get("columnNames") or []]
    return [dict(zip(columns, row, strict=False)) for row in table.get("rows") or []]


def _as_list(value: Any) -> list[str]:
    if value in (None, "", ()):
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def safe_name(text: Any) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(text)) or "unknown"
