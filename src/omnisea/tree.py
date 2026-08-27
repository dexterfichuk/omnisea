"""Assembling fetched series into an :class:`xarray.DataTree`, and reading one back out.

The tree is the answer to the doc's central problem: a 1-D tide gauge series and a 4-D model grid
cannot share one flat table without a lossy join, but they sit side by side in a tree perfectly
well. Each station becomes a group; nothing is resampled, reindexed or merged to fit.

The return value is a **plain** ``DataTree``, not a subclass, so it interoperates with everything
in the xarray ecosystem. Conveniences are module-level functions instead of methods.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .providers.base import StationSeries
from .query import Query

__all__ = [
    "build_tree",
    "data_nodes",
    "fields",
    "scalar_coord",
    "series_to_dataset",
    "summary",
    "stations",
    "to_dataframe",
    "to_netcdf",
    "coverage",
    "describe",
    "query_attrs",
]

log = logging.getLogger("omnisea.tree")

#: Columns :func:`summary` always produces, so an empty result is still addressable.
SUMMARY_COLUMNS = [
    "node", "provider", "site", "station_id", "station_name", "lat", "lon",
    "variables", "n_time", "start", "end",
]

#: Columns :func:`fields` always produces, for the same reason.
FIELD_COLUMNS = [
    "variable", "standard_name", "units", "long_name", "cf_mapped", "kind",
    "source_field", "cell_methods", "provider", "source", "station_id", "node", "n_values",
]


def _clean_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce attributes into something netCDF can actually store.

    netCDF attributes may only be strings or numeric arrays. Anything else — ``None``, a dict, a
    Timestamp — is converted rather than dropped, so that ``tree.to_netcdf()`` round-trips
    without the caller having to sanitize first.
    """
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, (str, int, float, np.number)):
            out[key] = value
        elif isinstance(value, (pd.Timestamp, datetime)):
            out[key] = pd.Timestamp(value).isoformat()
        elif isinstance(value, Mapping):
            out[key] = json.dumps(value, default=str)
        elif isinstance(value, (list, tuple, np.ndarray)):
            seq = list(value)
            if seq and all(isinstance(v, (int, float, np.number)) for v in seq):
                out[key] = [float(v) for v in seq]
            else:
                out[key] = ", ".join("" if v is None else str(v) for v in seq)
        else:
            out[key] = str(value)
    return out


def _netcdf_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Make a frame's dtypes survive a netCDF round-trip without changing its meaning.

    Three conversions, none of them optional if ``tree.to_netcdf()`` is to work natively:

    * **tz-aware -> naive UTC index.** CF expresses the timezone in the ``units`` string, and
      xarray's netCDF writer rejects ``datetime64[us, UTC]`` outright. Everything in omnisea is
      UTC by construction, so dropping the tzinfo loses nothing and the ``time`` coordinate says
      so in its attributes.
    * **bool -> int8.** netCDF has no boolean type; the values keep their flag meaning.
    * **object -> str.** Mixed str/NaN object columns cannot be encoded; nulls become empty
      strings, which is the conventional netCDF fill for a character variable.
    """
    frame = frame.copy()
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
        frame.index.name = "time"
    if isinstance(frame.index, pd.DatetimeIndex):
        # One time resolution everywhere. griddap nodes come back datetime64[ns] while point
        # nodes were datetime64[us], and pandas' merge_asof refuses to join across resolutions
        # — so two nodes of the same tree could not be merged in plain pandas.
        frame.index = frame.index.as_unit("ns")
    for col in frame.columns:
        kind = frame[col].dtype
        if pd.api.types.is_bool_dtype(kind):
            frame[col] = frame[col].astype("int8")
        elif pd.api.types.is_object_dtype(kind):
            frame[col] = frame[col].where(frame[col].notna(), "").astype(str)
    return frame


def series_to_dataset(series: StationSeries) -> xr.Dataset:
    """Turn one :class:`StationSeries` into a CF discrete-sampling-geometry ``Dataset``.

    Unit conversion is *not* done here — it happens where the provider reads the values, so the
    ``units`` attribute and the numbers beside it can never disagree.

    Station identity lands as *scalar coordinates* rather than attributes, which is what makes
    ``xr.concat`` across stations work later and what CF expects of a ``timeSeries`` feature.
    """
    frame = series.frame
    match = series.match

    if frame is not None and not frame.empty and isinstance(frame.index, pd.DatetimeIndex):
        if frame.index.tz is None:
            # Every source that goes through frame_from_records arrives tz-aware. One building
            # its index by hand does not, and a network publishing local wall-clock times then
            # ships every timestamp shifted by its own UTC offset with the values unchanged —
            # which align() will happily join against a correct source. Say so: this is the
            # single likeliest mistake for any provider outside UTC, and it is invisible.
            message = (
                f"{series.match.source}: node {series.node_path} has timezone-naive timestamps, "
                "which omnisea reads as UTC. If the upstream publishes local time, every "
                "timestamp is shifted by its offset. Parse with utc=True (or use "
                "frame_from_records, which does)."
            )
            warnings.warn(message, UserWarning, stacklevel=2)
            log.warning("%s", message)

    if frame is None or frame.empty:
        ds = xr.Dataset()
    else:
        ds = xr.Dataset.from_dataframe(_netcdf_safe_frame(frame))
        if "time" in ds.coords:
            ds["time"].attrs.update(
                {
                    "standard_name": "time",
                    "long_name": "time",
                    "axis": "T",
                    "time_zone": "UTC",
                    "comment": "All omnisea times are UTC; stored naive per CF convention.",
                }
            )

    ds = ds.assign_coords(
        latitude=float(match.lat),
        longitude=float(match.lon),
        station_id=str(match.station_id),
        station_name=str(match.name or match.station_id),
    )
    ds["latitude"].attrs.update(
        {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"}
    )
    ds["longitude"].attrs.update(
        {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}
    )
    ds["station_id"].attrs.update({"long_name": "station identifier", "cf_role": "timeseries_id"})
    ds["station_name"].attrs.update({"long_name": "station name"})

    if match.site:
        # The requested site this station answers for — the join key for multi-site queries.
        ds = ds.assign_coords(site=str(match.site))
        ds["site"].attrs.update({"long_name": "requested site label"})

    for var, attrs in series.var_attrs.items():
        if var in ds.variables:
            ds[var].attrs.update(_clean_attrs(attrs))

    node_attrs = dict(series.attrs)
    if match.distance_km is not None:
        node_attrs.setdefault("distance_from_site_km", round(match.distance_km, 3))
    ds.attrs.update(_clean_attrs(node_attrs))
    return ds


def build_tree(
    query: Query,
    results: Iterable[StationSeries | xr.Dataset],
    *,
    group_by_site: bool = False,
    drop_empty: bool = True,
) -> xr.DataTree:
    """Assemble fetched results into a single tree.

    ``group_by_site=True`` nests every node beneath the requested site it belongs to
    (``/<site>/in_situ/tides/07120``), which is the convenient shape when you handed omnisea a
    list of locations and want to address results by location.

    Stations that returned no rows in the window are dropped by default — a station whose record
    simply does not cover the requested dates should not appear as an empty group.
    """
    nodes: dict[str, xr.Dataset] = {}
    empty: list[str] = []
    contributed: set[str] = set()

    for item in results:
        if item is None:
            continue
        if isinstance(item, xr.Dataset):
            # The gridded path: a provider handed us a ready dataset with its own node path.
            path = str(item.attrs.get("omnisea_node_path") or "gridded/unnamed")
            nodes[_unique(path, nodes)] = item
            continue

        if drop_empty and item.is_empty:
            # Naming them matters. A station can be discovered and still hold nothing for the
            # window — ECCC's catalogue overstates several stations' periods of record — and
            # with nearest=1 that means the closest station silently yields an empty tree while
            # one a few km further has decades.
            empty.append(f"{item.match.source}/{item.match.station_id}")
            log.info(
                "no rows for %s station %s in the requested window",
                item.match.source,
                item.match.station_id,
            )
            continue

        contributed.add(f"{item.match.source}/{item.match.station_id}")
        ds = series_to_dataset(item)
        path = item.node_path.strip("/")
        if group_by_site and item.match.site:
            path = f"{_slug(item.match.site)}/{path}"
        nodes[_unique(path, nodes)] = ds

    root_attrs = _clean_attrs(
        {
            **query.to_attrs(),
            "title": "omnisea data tree",
            "Conventions": "CF-1.10",
            "institution": "assembled by omnisea",
            "source": "omnisea multi-provider retrieval",
            "history": (
                f"{datetime.now(UTC).isoformat(timespec='seconds')} "
                f"created by omnisea {_version()}"
            ),
            "omnisea_version": _version(),
            "n_nodes": len(nodes),
            "n_empty_series_dropped": len(empty),
            # Only stations that contributed NOTHING from that source. A station can serve one
            # series and lack another — DFO's Pruth Bay gauge has observed water levels but no
            # harmonic predictions — and charging the empty second series to the station put
            # the principal instrument of a whole analysis in the "returned no rows" list.
            # citation() prints this list precisely so a partial pull cannot pass unnoticed; a
            # false alarm on the most-used station teaches people to ignore it.
            "omnisea_empty_stations": sorted(set(empty) - contributed) or None,
        }
    )

    tree = xr.DataTree.from_dict({f"/{p}": ds for p, ds in nodes.items()})
    tree.attrs.update(root_attrs)
    return tree


def _unique(path: str, existing: Mapping[str, Any]) -> str:
    """Never let one node silently overwrite another."""
    if path not in existing:
        return path
    i = 2
    while f"{path}_{i}" in existing:
        i += 1
    return f"{path}_{i}"


def _slug(text: str) -> str:
    """A node-name-safe version of a user's site label."""
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(text))
    return safe.strip("_") or "site"


def _version() -> str:
    from . import __version__

    return __version__


# --------------------------------------------------------------------------- readers


def data_nodes(tree: xr.DataTree):
    """Yield ``(path, dataset)`` for every group that actually holds variables.

    The tree readers here and in :mod:`omnisea.provenance` are all built on this, and it is
    public because "walk the nodes that hold data" is the natural first step of any custom
    analysis over a fetched tree.
    """
    for node in tree.subtree:
        ds = node.dataset
        if len(ds.data_vars) == 0:
            continue
        yield node.path, ds


def summary(tree: xr.DataTree) -> pd.DataFrame:
    """One row per node: where it came from, what is in it, and over what period.

    This is the "did I get what I asked for?" view, and the first thing to print after a fetch.
    """
    rows: list[dict[str, Any]] = []
    for path, ds in data_nodes(tree):
        data_vars = [v for v in ds.data_vars if not str(v).endswith("_qc")]
        times = ds["time"].values if "time" in ds.coords and ds["time"].size else np.array([])
        rows.append(
            {
                "node": path,
                "provider": ds.attrs.get("provider", ""),
                "site": scalar_coord(ds, "site"),
                "station_id": scalar_coord(ds, "station_id"),
                "station_name": scalar_coord(ds, "station_name"),
                "lat": scalar_coord(ds, "latitude"),
                "lon": scalar_coord(ds, "longitude"),
                "variables": ", ".join(str(v) for v in data_vars),
                "n_time": int(times.size),
                "start": pd.Timestamp(times.min()) if times.size else pd.NaT,
                "end": pd.Timestamp(times.max()) if times.size else pd.NaT,
            }
        )
    # Named columns even when empty: an empty tree is exactly when someone reaches for the
    # "did I get what I asked for?" view, and a 0x0 frame turns summary(tree)["node"] into a
    # KeyError instead of an empty answer.
    frame = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["provider", "node"]).reset_index(drop=True)
    return frame


def describe(tree: xr.DataTree, width: int = 46) -> pd.DataFrame:
    """The at-a-glance view of a result: what you got, from where, over what period.

    :func:`summary` returns all eleven columns, which pandas elides to ``node ... end`` in a
    terminal — the columns a reader wants are exactly the ones that disappear. This is the
    same information, narrowed to fit and with long variable lists trimmed.
    """
    frame = summary(tree)
    if frame.empty:
        return frame
    out = frame[["node", "station_name", "n_time", "start", "end", "variables"]].copy()
    out["variables"] = out["variables"].map(
        lambda v: v if not isinstance(v, str) or len(v) <= width else v[: width - 1] + "\u2026"
    )
    for column in ("start", "end"):
        out[column] = pd.to_datetime(out[column]).dt.strftime("%Y-%m-%d %H:%M")
    return out


def stations(tree: xr.DataTree) -> pd.DataFrame:
    """Distinct stations present in the tree, one row each."""
    frame = summary(tree)
    if frame.empty:
        return pd.DataFrame(
            columns=["provider", "station_id", "station_name", "site", "lat", "lon",
                     "n_nodes", "n_time"]
        )
    return (
        # dropna=False: a gridded node has no station id, and pandas drops null group keys by
        # default — so a tree whose entire content was a model reported zero stations, and
        # auditing a saved file that way says the model was never fetched.
        frame.groupby(["provider", "station_id"], as_index=False, dropna=False)
        .agg(
            station_name=("station_name", "first"),
            site=("site", "first"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            n_nodes=("node", "count"),
            n_time=("n_time", "sum"),
        )
        .sort_values(["provider", "station_id"])
        .reset_index(drop=True)
    )


def to_dataframe(tree: xr.DataTree, *, wide: bool = False) -> pd.DataFrame:
    """Flatten the tree into one tidy frame.

    Long by default — ``time, site, provider, station_id, variable, value`` — because that is the
    shape that survives stations having different variables and different sampling rates. The
    tree remains the lossless container; this is the convenience view for plotting and joins.
    """
    pieces: list[pd.DataFrame] = []
    for path, ds in data_nodes(tree):
        if "time" not in ds.coords or ds["time"].size == 0:
            # A profile keyed on depth rather than time has real values that this flat,
            # time-indexed view cannot represent. Saying so beats returning fewer rows than
            # summary() and fields() both report — align() logs the same condition.
            log.warning(
                "to_dataframe(): skipping node %s — no 'time' coordinate, so its %d variable(s) "
                "cannot be placed on a time axis. Read it directly with tree[%r].",
                path,
                len(ds.data_vars),
                path,
            )
            continue
        keep = [v for v in ds.data_vars if not str(v).endswith("_qc")]
        if not keep:
            continue
        frame = ds[keep].to_dataframe().reset_index()
        for col in ("latitude", "longitude", "station_id", "station_name", "site"):
            if col in ds.coords and col not in frame.columns:
                frame[col] = scalar_coord(ds, col)
        frame["node"] = path
        frame["provider"] = ds.attrs.get("provider", "")
        pieces.append(frame)

    if not pieces:
        return pd.DataFrame(
            columns=["time", "site", "provider", "station_id", "variable", "value"]
        )

    joined = pd.concat(pieces, ignore_index=True)
    id_cols = [
        c
        for c in ("time", "node", "provider", "site", "station_id", "station_name",
                  "latitude", "longitude")
        if c in joined.columns
    ]
    value_cols = [c for c in joined.columns if c not in id_cols]

    if wide:
        return joined.sort_values("time").reset_index(drop=True)

    long = joined.melt(
        id_vars=id_cols, value_vars=value_cols, var_name="variable", value_name="value"
    )
    long = long.dropna(subset=["value"])
    return long.sort_values(["time", "provider", "station_id", "variable"]).reset_index(drop=True)


def fields(tree: xr.DataTree, *, mapped: bool | None = None) -> pd.DataFrame:
    """Every variable actually returned, and what omnisea knows about each one.

    ``omnisea.variables()`` lists what the curated CF tables *can* name; this lists what a
    particular fetch really brought back, which is usually more. Platforms add channels, and
    omnisea carries anything it has no CF mapping for under the provider's own field name — so
    this is the honest inventory of a result.

    ``mapped=True`` shows only the CF-described variables, ``mapped=False`` only the
    carried-through ones, and the default shows both.
    """
    rows: list[dict[str, Any]] = []
    for path, ds in data_nodes(tree):
        for name, var in ds.data_vars.items():
            name = str(name)
            is_qc = name.endswith("_qc")
            attrs = var.attrs
            is_mapped = bool(attrs.get("standard_name")) or attrs.get("omnisea_mapped") == 1
            rows.append(
                {
                    "variable": name,
                    "standard_name": attrs.get("standard_name", ""),
                    "units": attrs.get("units", ""),
                    "long_name": attrs.get("long_name", ""),
                    "cf_mapped": bool(is_mapped) and not is_qc,
                    "kind": "qc flag" if is_qc else ("CF" if is_mapped else "carried"),
                    "source_field": attrs.get("source_field", ""),
                    "cell_methods": attrs.get("cell_methods", ""),
                    "provider": ds.attrs.get("provider", ""),
                    "source": ds.attrs.get("source_name", ""),
                    "station_id": scalar_coord(ds, "station_id"),
                    "node": path,
                    "n_values": _count_if_loaded(var),
                }
            )
    frame = pd.DataFrame(rows, columns=FIELD_COLUMNS)
    if frame.empty:
        return frame
    if mapped is True:
        frame = frame[frame["cf_mapped"]]
    elif mapped is False:
        frame = frame[~frame["cf_mapped"] & (frame["kind"] != "qc flag")]
    return frame.sort_values(["node", "kind", "variable"]).reset_index(drop=True)


def query_attrs(tree: xr.DataTree) -> dict[str, Any]:
    """The query that produced this tree, decoded back into Python values.

    The root records the query as netCDF-safe attributes — JSON strings for the per-site
    arrays, since netCDF writes a one-element array as a scalar and readers would otherwise
    have to handle both shapes. This decodes them.
    """
    out: dict[str, Any] = {}
    for key, value in tree.attrs.items():
        if not str(key).startswith(("query_", "omnisea_")):
            continue
        if isinstance(value, str) and value[:1] in "[{":
            try:
                out[key] = json.loads(value)
                continue
            except ValueError:
                pass
        out[key] = value
    return out


def coverage(tree: xr.DataTree, query: Query | None = None) -> pd.DataFrame:
    """Which requested sites actually got data — including the ones that got none.

    When you hand omnisea a long list of locations, the sites that came back *empty* are the
    result you most need to see; a summary that only lists successes quietly hides the gaps.
    """
    frame = summary(tree)
    if frame.empty:
        got = pd.DataFrame(columns=["site", "n_nodes", "n_time", "providers"])
    else:
        got = (
            frame.assign(site=frame["site"].fillna(""))
            .groupby("site", as_index=False)
            .agg(
                n_nodes=("node", "count"),
                n_time=("n_time", "sum"),
                providers=("provider", lambda s: ", ".join(sorted(set(s)))),
            )
        )

    requested = _requested_sites(tree, query)
    if not requested:
        return got.reset_index(drop=True)

    got = got.set_index("site")
    rows = []
    for label in requested:
        if label in got.index:
            row = got.loc[label]
            rows.append(
                {
                    "site": label,
                    "n_nodes": int(row["n_nodes"]),
                    "n_time": int(row["n_time"]),
                    "providers": row["providers"],
                    "has_data": bool(row["n_time"] > 0),
                }
            )
        else:
            rows.append(
                {"site": label, "n_nodes": 0, "n_time": 0, "providers": "", "has_data": False}
            )
    return pd.DataFrame(rows)


def _requested_sites(tree: xr.DataTree, query: Query | None) -> list[str]:
    if query is not None and query.sites:
        return [s.label for s in query.sites]
    names = tree.attrs.get("query_site_names")
    if isinstance(names, str):
        try:
            decoded = json.loads(names)
        except ValueError:
            # A tree written before labels were JSON-encoded. Comma-splitting is what broke
            # them, so treat the whole string as one label rather than inventing several.
            return [names.strip()] if names.strip() else []
        return [str(n) for n in decoded] if isinstance(decoded, list) else []
    if isinstance(names, Sequence):
        return [str(n) for n in names]
    return []


def _count_if_loaded(var: xr.DataArray) -> int | None:
    """Non-null count, but never at the cost of a download.

    Counting is cheap for a station series and is a silent transfer for a lazy gridded
    variable, which is the whole point of keeping those lazy. ``None`` means "not counted".
    """
    if var.size == 0:
        return 0
    if var.chunks is not None or not var.variable._in_memory:
        return None
    return int(var.count().values)


def scalar_coord(ds: xr.Dataset, name: str) -> Any:
    """A scalar coordinate's value, or ``None`` if the coordinate is not scalar.

    A gridded node's ``latitude`` is a whole vector; returning it would put the entire array in
    a table cell.
    """
    if name not in ds.coords:
        return None
    coord = ds[name]
    if coord.size != 1:
        return None
    value = coord.values
    try:
        item = value.item()
    except (ValueError, AttributeError):
        return value
    return item


def to_netcdf(tree: xr.DataTree, path: Any, **kwargs: Any) -> None:
    """Write a tree to netCDF, naming the extra to install if no engine is present.

    ``tree.to_netcdf(path)`` works exactly as well and stays the documented way to do this.
    This wrapper exists for the one thing xarray cannot know: that the missing backend is
    available as ``pip install "omnisea[netcdf]"``. Without it a user following the README on a
    bare install gets a raw xarray error naming two libraries they never asked for.
    """
    import importlib.util

    if not any(importlib.util.find_spec(name) for name in ("netCDF4", "h5netcdf")):
        from .errors import MissingDependencyError

        raise MissingDependencyError("netCDF4", "netcdf", "for writing netCDF files")
    lazy_cells = 0
    lazy_nodes: list[str] = []
    for node_path, ds in data_nodes(tree):
        for variable in ds.data_vars.values():
            if not getattr(variable.variable, "_in_memory", True):
                lazy_cells += int(variable.size)
                lazy_nodes.append(node_path)
                break
    if lazy_cells > 10_000_000:
        # Writing materializes every lazy grid in the tree — "you pay only for the pixels you
        # read" ends at to_netcdf(). A tree holding a subset model field wrote 34 MB without
        # comment; the full physics node would have quietly attempted ~237 GB. Say the price
        # before charging it.
        import warnings

        warnings.warn(
            f"to_netcdf() will download and write every lazy grid in this tree — about "
            f"{lazy_cells:,} cells across {len(lazy_nodes)} node(s) ({', '.join(lazy_nodes[:3])}"
            f"{'…' if len(lazy_nodes) > 3 else ''}). Subset first (ds.isel/sel) or drop the "
            "grid nodes (del tree[path]) if that is more than you meant to store.",
            UserWarning,
            stacklevel=2,
        )
    tree.to_netcdf(path, **kwargs)
