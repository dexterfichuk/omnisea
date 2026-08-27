"""``griddap`` — gridded fields, returned lazily over OPeNDAP.

The subset is expressed as an :meth:`xarray.Dataset.sel` on the open remote dataset, so no
bytes move until the caller actually indexes the result.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from contextlib import contextmanager
from typing import Any

import xarray as xr

from ... import cf
from ...errors import ProviderError, UpstreamError
from ...query import BBox, Query
from ..base import StationMatch
from .common import ErddapSource, safe_name
from .info import LATITUDE_NAMES, LONGITUDE_NAMES, TIME_NAMES

log = logging.getLogger("omnisea.erddap")

__all__ = ["ErddapGridSource", "grid_selection"]


class ErddapGridSource(ErddapSource):
    """``griddap`` — gridded fields, returned lazily over OPeNDAP.

    The subset is expressed as an :meth:`xarray.Dataset.sel` on the open remote dataset, so no
    bytes move until the caller indexes the result. That is the point of the gridded path: a user
    can put a decade of a global SST analysis in their tree and only pay for the pixels they read.
    """

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
        selection = grid_selection(remote, query)
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
                omnisea_node_path=f"{self.branch_for(match)}/{safe_name(dataset_id)}",
                omnisea_cells_estimate=cells,
                site=match.site,
            ),
        }
        return subset

    @staticmethod
    def _open(url: str) -> xr.Dataset:
        """Open an ERDDAP griddap endpoint as a lazy dataset.

        This is the one request in the package that does not go through :mod:`omnisea.http`: DAP
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
            with _quiet_hdf5_stderr():
                return xr.open_dataset(url, chunks=chunks, decode_timedelta=False)
        except Exception as exc:  # noqa: BLE001 - surfaced as an omnisea error
            raise UpstreamError(
                f"could not open griddap dataset: {exc}", provider="erddap_griddap", url=url
            ) from exc


# --------------------------------------------------------------------------- grid subsetting


def grid_selection(dataset: xr.Dataset, query: Query) -> dict[str, slice]:
    """The lazy ``sel`` that clips a griddap dataset to the query, in each axis's own order.

    Grids disagree about direction and about longitude convention — latitude descends as often as
    it ascends, and a global grid may run 0-360 — so each slice is built from the coordinate's
    own values rather than from the query's orientation.
    """
    selection: dict[str, slice] = {}

    time_name = _axis_name(dataset, "T", TIME_NAMES)
    if time_name is not None:
        # griddap times decode to naive datetime64; omnisea's are tz-aware UTC.
        selection[time_name] = _ordered_slice(
            dataset[time_name], query.start.tz_localize(None), query.end.tz_localize(None)
        )

    if query.bbox is not None:
        lat_name = _axis_name(dataset, "Y", LATITUDE_NAMES)
        if lat_name is not None:
            selection[lat_name] = _ordered_slice(
                dataset[lat_name], query.bbox.south, query.bbox.north
            )
        lon_name = _axis_name(dataset, "X", LONGITUDE_NAMES)
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


@contextmanager
def _quiet_hdf5_stderr() -> Any:
    """Silence the HDF5 error stack a *successful* DAP open prints.

    netCDF-C probes the URL as a local file before falling back to the DAP client, and HDF5
    prints a ~2.3 KB "unable to open file" stack to C-level stderr on that probe — on opens
    that then succeed. It lands in notebook cells and reads as a failure. Python-level
    redirection cannot catch it (it is written by C), so the OS file descriptor is swapped for
    the duration of the open and restored in all cases. Skipped when stderr has no real fd
    (some embedded interpreters), where the noise cannot appear anyway.
    """
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    saved = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, stderr_fd)
        os.close(saved)
        os.close(devnull)
