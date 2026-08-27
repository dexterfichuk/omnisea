"""Behaviour the ``tabledap`` and ``griddap`` halves of an ERDDAP server share.

Discovery is the bulk of it: both protocols find candidate datasets the same way (the
``allDatasets`` table and the search index, unioned), read the same ``/info`` metadata, apply
the same extent and time filters, and answer "nothing matched" the same odd way (HTTP 404).
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

import pandas as pd

from ... import cf
from ...errors import (
    OmniseaError,
    PayloadTooLargeError,
    ProviderError,
    QueryError,
    UpstreamError,
)
from ...http import DEFAULT_MAX_WORKERS, get_json, map_threads
from ...query import BBox, Query, register_option
from ..base import RetrievalSource, StationMatch
from .info import DatasetInfo, cached_info, parse_info, store_info
from .servers import ErddapServer, resolve_servers, server_name_for_url

log = logging.getLogger("omnisea.erddap")

__all__ = [
    "DEFAULT_SERVER",
    "ErddapServer",
    "DEFAULT_MAX_DATASETS",
    "ErddapSource",
    "table_rows",
    "safe_name",
]

register_option(
    "erddap_server",
    "erddap: server name(s) or root URL(s) to query; 'all' sweeps every known "
    "installation (default IOOS Sensors). See omnisea.erddap_servers().",
)
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

    def servers(self, query: Query) -> list[ErddapServer]:
        """Which installations this query asks for — one, several, or every known one.

        ERDDAP's reach is its whole point, and a user cannot query a server they have never
        heard of. ``erddap_server=`` therefore takes a short name as readily as a URL, a list
        of either, or ``"all"``.
        """
        return resolve_servers(query.option("erddap_server"), self.provider.base_url)

    def server(self, query: Query) -> str:
        """The single server this query names, for the paths that can only mean one.

        Kept because a source that has already resolved a match reads the server off the match
        itself; this is only for the callers that ask before discovery has run.
        """
        chosen = self.servers(query)
        if len(chosen) > 1:
            raise QueryError(
                f"this query names {len(chosen)} ERDDAP servers "
                f"({', '.join(s.name for s in chosen)}) and this call needs exactly one"
            )
        return chosen[0].url

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
        chosen = self.servers(query)
        if len(chosen) == 1:
            return self._discover_one(query, chosen[0])

        # Several installations. One of them being down, slow or mid-reindex must not cost the
        # caller the others -- these are a dozen independent institutions, and requiring all of
        # them to be healthy would make the sweep less reliable the more of it you use. What
        # failed is recorded on the source's notes, so an incomplete sweep still says so.
        results = map_threads(
            lambda s: self._discover_safely(query, s),
            chosen,
            max_workers=int(query.option("max_workers", DEFAULT_MAX_WORKERS)),
            label=f"{self.name} servers",
        )
        matches: list[StationMatch] = []
        failed: list[str] = []
        for server, found, error in results:
            if error is not None:
                failed.append(f"{server.name}: {error}")
            matches.extend(found)
        if len(failed) == len(chosen):
            # Every one of them. Zero matches would read as "there is nothing at this place",
            # which is a different answer from "nobody was reachable to ask".
            raise UpstreamError(
                f"none of the {len(chosen)} ERDDAP servers answered: " + "; ".join(failed),
                provider=self.name,
            )
        absent_only = all("does not host the dataset(s) named" in f for f in failed)
        if failed and matches and absent_only:
            failed = []
        if failed:
            note = (
                f"reached {len(chosen) - len(failed)} of {len(chosen)} ERDDAP servers; "
                f"no answer from {'; '.join(failed)}"
            )
            self._notes.value = note
            log.warning("%s: %s", self.name, note)
        return matches

    #: Set by a partial sweep and read by the thread that ran it. Thread-local because one
    #: source object is shared by every query in the process, and discovery runs each source in
    #: its own thread — a plain attribute would let one query's note surface in another's.
    _notes = threading.local()

    def branch_for(self, match: StationMatch) -> str:
        """This match's node branch: the source's own, then the installation it came from.

        A dataset id identifies a dataset *on one server* and nothing more — DFO's gliders are
        published on CIOOS Pacific and on the IOOS Glider DAC under identical ids. Naming the
        server in the path is what keeps two of them from landing on top of each other, and
        makes a path say where its data is from without opening the node.
        """
        server = str(match.extra.get("server_name") or "")
        if not server:
            # A match built without discovery having named the server -- a hand-made one, or a
            # tree reopened from a file. The URL still identifies it, so the path a dataset
            # lands at does not depend on how the match was made.
            server = server_name_for_url(str(match.extra.get("server") or ""))
        return f"{self.node_path}/{safe_name(server)}"

    def take_discovery_note(self) -> str | None:
        note = getattr(self._notes, "value", None)
        self._notes.value = None
        return note

    def _discover_safely(
        self, query: Query, server: ErddapServer
    ) -> tuple[ErddapServer, list[StationMatch], str | None]:
        try:
            return server, self._discover_one(query, server), None
        except UpstreamError as exc:
            if exc.status == 404 and "unknown datasetID" in (exc.detail or ""):
                # erddap_datasets= named something this installation does not host. Across a
                # sweep that is the expected answer from every server but the one that has it,
                # so it is not a failure. If no server has it, the all-failed check below still
                # raises and names them.
                return server, [], "does not host the dataset(s) named"
            return server, [], f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
        except PayloadTooLargeError as exc:
            # The dataset ceiling is per-server and exists to stop omnisea describing hundreds
            # of datasets at one institution. Letting it end the whole sweep would mean that
            # adding a server to the list could *reduce* what you get back -- so this one is
            # reported and skipped like any other, but named so it can be raised deliberately.
            return server, [], (
                f"over the dataset ceiling ({exc.estimate} datasets match); narrow the "
                "query, name datasets with erddap_datasets=[...], or raise erddap_max_datasets"
            )
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            return server, [], f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"

    def _discover_one(self, query: Query, server_info: ErddapServer) -> list[StationMatch]:
        server = server_info.url
        named = _as_list(query.option("erddap_datasets"))
        candidates = named or self._candidate_ids(query, server)
        if not candidates:
            # Nothing the area filter could reach. That may be true, or it may be that the
            # publisher declared no extent — ask before letting an absence read as an answer.
            self._note_extentless(server)
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

        workers = int(query.option("max_workers", DEFAULT_MAX_WORKERS))
        if named:
            # erddap_datasets= is one list for the whole query, so it can legitimately name a
            # tabledap station, a griddap grid and a dataset that lives on a different server.
            # Failing the batch because one id is not on *this* installation costs the caller
            # every other source in the query -- a satellite grid and two US stations vanished
            # from an otherwise successful five-source fetch that way. Resolve them one at a
            # time and keep what this server actually has.
            resolved = map_threads(
                lambda ds: self._info_or_absent(server, ds),
                candidates,
                max_workers=workers,
                label=f"{self.name} dataset info",
            )
            infos = [info for info in resolved if info is not None]
            if not infos:
                raise UpstreamError(
                    f"none of the dataset(s) named are on {server}: "
                    + ", ".join(map(str, candidates)),
                    provider=self.name,
                    status=404,
                    detail="Currently unknown datasetID",
                )
            absent = len(candidates) - len(infos)
            if absent:
                log.debug(
                    "%s: %d of %d named dataset(s) are not on %s",
                    self.name, absent, len(candidates), server,
                )
        else:
            infos = map_threads(
                lambda ds: self._info(server, ds),
                candidates,
                max_workers=workers,
                label=f"{self.name} dataset info",
            )

        wanted = cf.resolve_names(query.variables)
        matches: list[StationMatch] = []
        for info in infos:
            unusable = self.unusable_reason(info)
            if (
                unusable
                and named
                and self._wrong_protocol_reason(info)
                and self._sibling_selected(query)
            ):
                # A named list can legitimately mix a table and a grid -- one erddap_datasets=
                # serves the whole query. The sibling source is in this query and will take it,
                # so refusing here would cost the caller the datasets this source *can* read.
                # Only the wrong-protocol case: a dataset with no time axis at all is unusable
                # by both halves, and skipping that one would just be silence.
                log.debug("%s: leaving %s to the other protocol", self.name, info.dataset_id)
                continue
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
                # Dataset ids are unique per installation, not across them: DFO's gliders
                # appear on CIOOS Pacific and on the IOOS Glider DAC under the same id. The
                # server is what makes a node path mean one thing.
                match.extra["server_name"] = server_info.name
                # _match_for already attached the site and, for datasets with real extent,
                # corrected the distance to the nearest edge. Re-attaching would undo that.
                matches.append(match)
        if not named and not matches:
            self._note_extentless(server)
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
        # A *point* extent is a fixed station and its centre is its position. Anything wider is
        # a track, a grid or a fleet, and the middle of its bounding box is not where the data
        # is: ArgoFloats' box spans the globe, so its "position" came out in Nebraska, 12,982 km
        # from the query, in a column sitting beside real station coordinates. One CIOOS
        # Atlantic dataset declares longitude -521.310. The position is unknown until the rows
        # arrive, so say unknown; fetch fills it in from what actually comes back.
        if bounds is not None and _is_point(bounds):
            lat, lon = bounds.centre
        else:
            lat, lon = float("nan"), float("nan")
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
        from_search = self._from_search(query, server)
        if query.option("erddap_search"):
            # A free-text search is a request to NARROW. Unioning it with the unfiltered
            # catalogue meant erddap_search= could never remove anything: the same 62 datasets
            # came back with it, without it, and with a term matching nothing — and it is the
            # first thing anyone reaches for on hitting the erddap_max_datasets ceiling.
            # Only the search index knows about searchFor, so only it can answer this.
            return [d for d in from_search if d != "allDatasets"]
        found: list[str] = []
        for dataset_id in self._from_all_datasets(query, server) + from_search:
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

    def _note_extentless(self, server: str) -> None:
        """Say how many datasets no spatial filter could ever have reached.

        A publisher who omits ``geospatial_lat/lon_*`` makes their dataset invisible to every
        area query — ERDDAP drops it server-side, so omnisea never sees it to report it. Half
        of Hakai's catalogue is like this, and 25 of SalishSeaCast's 42 grids are (NEMO is
        curvilinear, indexed gridY/gridX, so its lat/lon extents really are NaN). The result was
        that at the Hakai Institute's own field station, on the server named after Hakai,
        discovery answered "no stations found" — an absence indistinguishable from an answer,
        which is the failure this library exists to prevent. Asked only when nothing matched,
        so the ordinary case costs no extra request.
        """
        try:
            # Built by hand, like _from_all_datasets: ERDDAP's variable list is a bare name
            # before the first constraint, which a params dict cannot express.
            payload = self._get(
                f"{server}/tabledap/allDatasets.json?datasetID"
                f'&dataStructure="{self.data_structure}"&minLongitude=NaN',
                None,
            )
        except OmniseaError:
            return
        hidden = len(table_rows(payload)) if payload else 0
        if hidden:
            self._notes.value = (
                f"nothing matched, but {hidden} {self.protocol} dataset(s) on {server} declare "
                "no geospatial extent, so no area query can reach them — they are excluded "
                "from this catalogue rather than absent from the server. Name them with "
                "erddap_datasets=[...] to fetch them anyway."
            )

    def _sibling_selected(self, query: Query) -> bool:
        """Is the other ERDDAP protocol also selected by this query?"""
        sibling = "erddap_griddap" if self.protocol == "tabledap" else "erddap_tabledap"
        asked = query.providers
        if not asked:
            return True  # an unqualified query runs every source, so both are in play
        names = [asked] if isinstance(asked, str) else list(asked)
        return sibling in names or self.provider.name in names

    def _info_or_absent(self, server: str, dataset_id: str) -> DatasetInfo | None:
        """``_info``, but ``None`` when this installation simply does not host the dataset."""
        try:
            return self._info(server, dataset_id)
        except UpstreamError as exc:
            if exc.status == 404 and "unknown datasetID" in (exc.detail or ""):
                return None
            raise
        except ProviderError:
            return None

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
        globals_ = info.global_attrs
        # A datum is not decoration. NOAA's water level is published above MLLW and DFO's above
        # chart datum; a tree holding both, with neither datum recorded, puts two sea-level
        # columns 0.64 m apart side by side with nothing to explain the difference. Whatever the
        # dataset says about its vertical reference travels with it.
        datum = str(globals_.get("vertical_datum") or globals_.get("geospatial_vertical_crs")
                    or "") or None
        return self.base_attrs(
            title=info.title,
            summary=str(globals_.get("summary") or "") or None,
            institution=info.institution or None,
            license=info.license or None,
            references=str(globals_.get("infoUrl") or globals_.get("license_url") or "") or None,
            datum=datum,
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
