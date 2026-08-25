"""Aligning heterogeneous series onto one time axis — the step before a model.

The tree is lossless but ragged: tides arrive every 15 minutes, climate summaries once a day,
tidal extrema at irregular turning points. A model wants a rectangle — one row per timestamp,
one column per variable. Getting there means resampling, and resampling a series wrongly is one
of the easiest ways to publish a wrong result.

omnisea does not have to guess how. CF ``cell_methods`` already says what each variable *is*:

* ``time: sum`` — an accumulation. Daily precipitation summed to weekly is right; averaged is
  wrong, and interpolating it to hourly invents a distribution across the day that was never
  measured. These are summed when downsampling and forward-filled when upsampling, never
  interpolated.
* ``time: maximum`` / ``time: minimum`` — extremes. The max of the maxima is a real maximum;
  their mean is not a statistic of anything.
* ``time: mean`` — an interval average. Safe to average when downsampling, but upsampled by
  forward-fill: a daily mean spread across its own day is honest, whereas interpolating between
  day centres invents intra-day structure that was never measured.
* no ``cell_methods`` at all, or ``time: point`` — an instantaneous reading such as a tide
  height. These are the only ones interpolated when upsampling.

So the aggregation is *derived from the metadata each provider already published*, and the
result records what it did per column, so the choice is auditable rather than implicit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .errors import QueryError

log = logging.getLogger("omnisea.align")

__all__ = [
    "align",
    "add_local",
    "aggregation_for",
    "correlations",
    "drop_correlated",
    "is_circular",
]

#: Units that mark an angular quantity.
DEGREE_UNITS = frozenset({"degree", "degrees", "deg", "degree_true", "degrees_true", "\u00b0"})

#: How each CF cell_method resamples: (downsample aggregation, upsample fill).
CELL_METHOD_RULES: dict[str, tuple[str, str]] = {
    "sum": ("sum", "ffill"),
    "maximum": ("max", "ffill"),
    "minimum": ("min", "ffill"),
    "mean": ("mean", "ffill"),
    "standard_deviation": ("mean", "ffill"),
    "point": ("mean", "interpolate"),
}

DEFAULT_RULE = ("mean", "interpolate")
CATEGORICAL_RULE = ("first", "ffill")

#: A compass direction cannot be averaged or interpolated as a number: 350 deg and 10 deg both
#: point nearly north, but their arithmetic mean is 180 deg -- due south -- and interpolating
#: between them sweeps the long way round through south. Directions are combined as unit
#: vectors instead, and never interpolated.
CIRCULAR_RULE = ("circular_mean", "ffill")


def is_circular(attrs: Mapping[str, Any], var: str = "") -> bool:
    """Is this variable an angle on a compass, rather than an ordinary number?

    Keyed on degree units *and* the name reading as a direction. Directional *spread* is
    excluded: a spread of 30 degrees is a width, which averages perfectly well.
    """
    units = str(attrs.get("units") or "").strip()
    if units not in DEGREE_UNITS:
        return False
    name = f"{attrs.get('standard_name') or ''} {var}".lower()
    return "direction" in name and "spread" not in name


def circular_mean(values: Any) -> float:
    """Mean compass bearing, via unit vectors. NaN for an empty window."""
    series = pd.Series(values).dropna().astype(float)
    if series.empty:
        return float("nan")
    radians = np.deg2rad(series.to_numpy())
    mean = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return float(np.rad2deg(mean) % 360.0)


def aggregation_for(
    attrs: Mapping[str, Any], *, numeric: bool = True, var: str = ""
) -> tuple[str, str]:
    """The ``(downsample, upsample)`` rule implied by a variable's CF attributes.

    Non-numeric variables (weather descriptions, flag strings) cannot be averaged, so they take
    the first value in a window and are forward-filled. Compass directions get their own rule --
    see :data:`CIRCULAR_RULE` for why an ordinary mean is not merely imprecise but backwards.
    """
    if not numeric:
        return CATEGORICAL_RULE
    if is_circular(attrs, var):
        return CIRCULAR_RULE
    cell_methods = str(attrs.get("cell_methods") or "")
    for token, rule in CELL_METHOD_RULES.items():
        if token in cell_methods:
            return rule
    return DEFAULT_RULE


def _node_frames(
    tree: xr.DataTree, *, include_qc: bool
) -> list[tuple[str, str, pd.DataFrame, dict[str, dict[str, Any]]]]:
    """Every data-bearing node as ``(station_label, node_path, frame, per-variable attrs)``."""
    out = []
    for node in tree.subtree:
        ds = node.dataset
        if not ds.data_vars:
            continue
        if "time" not in ds.coords:
            log.warning(
                "skipping node %s in align(): no 'time' coordinate (has %s)",
                node.path,
                ", ".join(map(str, ds.coords)) or "none",
            )
            continue
        if ds["time"].size == 0:
            continue
        keep = [
            str(v)
            for v in ds.data_vars
            if include_qc or not str(v).endswith("_qc")
        ]
        if not keep:
            continue
        frame = ds[keep].to_dataframe()
        # Scalar coords (latitude, station_id, ...) ride along as columns; drop them.
        frame = frame[[c for c in frame.columns if c in keep]]
        if not isinstance(frame.index, pd.DatetimeIndex):
            log.warning(
                "skipping node %s in align(): its time dimension is %r, not a DatetimeIndex "
                "named 'time'",
                node.path,
                frame.index.name,
            )
            continue
        label = _station_label(ds, node.path)
        attrs = {v: dict(ds[v].attrs) for v in keep}
        out.append((label, node.path, frame.sort_index(), attrs))
    return out


def _branch(node_path: str) -> str:
    """The node's parent path, used to tell two nodes of the same station apart."""
    parts = [p for p in node_path.strip("/").split("/") if p]
    return "_".join(parts[:-1]) or "node"


def _station_label(ds: xr.Dataset, path: str) -> str:
    for key in ("station_id", "site"):
        if key in ds.coords:
            try:
                return str(ds[key].values.item())
            except (ValueError, AttributeError):
                pass
    return path.strip("/").replace("/", "_")


def _normalize_target(on: Any) -> tuple[pd.DatetimeIndex, pd.DataFrame | None]:
    """Turn the user's own data into a target time index, keeping their columns if any.

    Naive timestamps are read as UTC, matching the rest of omnisea; tz-aware ones are converted.
    The tree stores naive UTC (CF puts the zone in the units), so the target must match.
    """
    carried: pd.DataFrame | None = None

    if isinstance(on, pd.DataFrame):
        frame = on.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            time_col = _find_time_column(frame)
            if time_col is None:
                raise QueryError(
                    "on= DataFrame needs a DatetimeIndex or a time column "
                    "(named time/datetime/date/timestamp)"
                )
            frame = frame.set_index(time_col)
        frame.index = _to_naive_utc(pd.DatetimeIndex(frame.index))
        frame = frame.sort_index()
        carried = frame
        index = frame.index
    elif isinstance(on, pd.Series):
        index = _to_naive_utc(pd.DatetimeIndex(pd.to_datetime(on.values)))
    elif isinstance(on, pd.DatetimeIndex):
        index = _to_naive_utc(on)
    elif isinstance(on, Iterable):
        index = _to_naive_utc(pd.DatetimeIndex(pd.to_datetime(list(on))))
    else:
        raise QueryError(f"could not interpret on={on!r} as timestamps")

    if index.has_duplicates:
        raise QueryError("on= timestamps must be unique")
    return index.sort_values(), carried


def _find_time_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("time", "datetime", "date", "timestamp", "obs_time"):
        for col in frame.columns:
            if str(col).lower() == candidate:
                return col
    return None


def _to_naive_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


def align(
    tree: xr.DataTree,
    *,
    freq: str | None = None,
    on: Any = None,
    tolerance: str | pd.Timedelta | None = None,
    direction: str = "nearest",
    columns: str = "auto",
    include_qc: bool = False,
    agg: Mapping[str, str | Callable[[Any], Any]] | None = None,
    interpolate: bool = True,
) -> pd.DataFrame:
    """Align every node onto one time axis and return a wide, model-ready frame.

    Give exactly one of:

    * ``freq`` — resample onto a regular grid, e.g. ``freq="1h"`` or ``freq="D"``.
    * ``on`` — align onto timestamps you already have: a DataFrame of your own observations
      (its columns are carried through, so you get ``y`` and ``X`` in one table), a Series, or
      a DatetimeIndex.

    With ``on``, how far a match may reach depends on what the variable is. Interval summaries
    (anything with ``cell_methods``, such as a daily total) match backwards within their own
    interval — the value for the day containing your sample. Instantaneous readings match to
    the nearest observation within ``tolerance``, defaulting to the series' own cadence.

    Each variable is resampled according to its CF ``cell_methods`` — sums stay sums, maxima
    stay maxima, instantaneous values may be interpolated. Override per variable with
    ``agg={"precipitation_amount": "mean"}`` if you really mean something else.

    ``columns`` controls naming when several stations serve the same variable:
    ``"auto"`` uses the bare name where it is unambiguous and appends ``@station`` where it is
    not, ``"qualified"`` always appends it (stable names for a pipeline, whatever the query
    returns), and ``"multi"`` gives a ``(variable, station)`` MultiIndex.

    The result carries ``frame.attrs["omnisea_aggregation"]``: the rule applied to each column,
    so the resampling choices are auditable rather than implicit.
    """
    if (freq is None) == (on is None):
        raise QueryError("give exactly one of freq= (a regular grid) or on= (your timestamps)")

    nodes = _node_frames(tree, include_qc=include_qc)
    overrides = dict(agg or {})

    if on is not None:
        target, carried = _normalize_target(on)
    else:
        target, carried = None, None

    grid = _grid_for(nodes, freq) if freq is not None else None

    pieces: list[pd.DataFrame] = []
    applied: dict[tuple[str, str], str] = {}

    used: set[tuple[str, str]] = set()
    for label, node_path, frame, attrs in nodes:
        for variable in frame.columns:
            series = frame[variable].dropna()
            if series.empty:
                continue
            numeric = pd.api.types.is_numeric_dtype(series)
            down, up = aggregation_for(
                attrs.get(variable, {}), numeric=numeric, var=str(variable)
            )
            if variable in overrides:
                down = overrides[variable]
                up = "ffill" if isinstance(down, str) and down in ("sum", "max", "min") else up
            if not interpolate and up == "interpolate":
                up = "ffill"

            if freq is not None:
                aligned, rule = _resample(series, freq, grid, down, up)
            else:
                interval = bool(str(attrs.get(variable, {}).get("cell_methods") or "").strip())
                aligned, rule = _join_to(series, target, tolerance, direction, interval)

            if aligned is None:
                continue
            # A station can appear in more than one branch (observed tides and predicted
            # extrema are both station 08545), so fall back to the branch name rather than
            # emitting two identically-named columns.
            key = (str(variable), label)
            if key in used:
                key = (str(variable), f"{label}/{_branch(node_path)}")
            used.add(key)
            aligned.name = key
            pieces.append(aligned.to_frame())
            applied[key] = rule

    if not pieces:
        if carried is not None:
            out = carried.copy()
            out.attrs["omnisea_carried"] = [str(c) for c in carried.columns]
            return out
        empty = pd.DataFrame(index=target if target is not None else pd.DatetimeIndex([]))
        empty.index.name = "time"
        return empty

    wide = pd.concat(pieces, axis=1)
    wide.index.name = "time"
    wide.columns = pd.MultiIndex.from_tuples(wide.columns, names=["variable", "station"])

    if carried is not None:
        own = carried.copy()
        own.columns = pd.MultiIndex.from_tuples(
            [(str(c), "") for c in own.columns], names=["variable", "station"]
        )
        wide = own.join(wide, how="left")

    wide = _name_columns(wide, columns)
    wide.attrs["omnisea_aggregation"] = {f"{v}@{s}" if s else v: r for (v, s), r in applied.items()}
    # Which columns are the caller's own rather than fetched features. drop_correlated() reads
    # this so it can never prune someone's response variable out of their own model matrix.
    wide.attrs["omnisea_carried"] = (
        [] if carried is None else [str(c) for c in carried.columns]
    )
    wide.attrs["omnisea_alignment"] = (
        f"resampled to {freq}" if freq else f"joined to {len(target)} supplied timestamps"
        + (f" within {tolerance}" if tolerance else "")
    )
    return wide


def _grid_for(nodes: list[tuple[str, str, pd.DataFrame, Any]], freq: str) -> pd.DatetimeIndex:
    """One time axis for every column in the call.

    Built by binning the overall span the way pandas itself would, so it is valid for calendar
    offsets ("ME", "QE") as well as fixed ones, and identical for every column — resampling each
    series independently could otherwise hand back frames that do not line up.
    """
    starts = [frame.index.min() for _, _, frame, _ in nodes if len(frame)]
    ends = [frame.index.max() for _, _, frame, _ in nodes if len(frame)]
    if not starts:
        return pd.DatetimeIndex([])
    # A single instant (or a single-point node) makes min == max; a duplicated label there
    # would propagate into the grid and break every reindex downstream.
    bounds = pd.DatetimeIndex([min(starts), max(ends)]).unique()
    span = pd.Series(0.0, index=bounds)
    return span.resample(freq).asfreq().index


def _resample(
    series: pd.Series, freq: str, grid: pd.DatetimeIndex, down: Any, up: str
) -> tuple[pd.Series | None, str]:
    """Put one series onto ``grid``, up- or down-sampling according to its own cadence."""
    if len(grid) == 0:
        return None, "empty grid"
    if len(series) < 2:
        return series.reindex(grid, method="ffill"), f"{down} (single sample)"

    native = series.index.to_series().diff().median()
    grid_step = pd.Series(grid).diff().median()

    if pd.notna(grid_step) and native > grid_step:
        # Upsampling. Reindex through the *union* of the series and the grid rather than
        # resampling: an irregular series (tidal extrema at 03:33, 09:58, ...) has no values on
        # the grid boundaries at all, and binning it would silently discard every one of them.
        combined = series.reindex(series.index.union(grid))
        filled = (
            combined.interpolate(method="time")
            if up == "interpolate"
            else combined.ffill()
        )
        return filled.reindex(grid), f"{up} (upsampled)"

    if down == "circular_mean":
        return series.resample(freq).apply(circular_mean).reindex(grid), "circular mean"

    try:
        return series.resample(freq).agg(down).reindex(grid), f"{down}"
    except (TypeError, ValueError, AttributeError):
        return series.resample(freq).first().reindex(grid), "first (not aggregatable)"


def _native_cadence(series: pd.Series) -> pd.Timedelta | None:
    if len(series) < 2:
        return None
    step = series.index.to_series().diff().median()
    return step if pd.notna(step) and step > pd.Timedelta(0) else None


def _join_to(
    series: pd.Series,
    target: pd.DatetimeIndex,
    tolerance: str | pd.Timedelta | None,
    direction: str,
    interval: bool,
) -> tuple[pd.Series | None, str]:
    """Join a series onto supplied timestamps, matching by what the variable actually is.

    An **interval summary** — anything carrying ``cell_methods``, such as a daily total — is
    matched *backwards* within its own interval: the value for the day that contains your
    sample. That is a containment question with a right answer, so it is not left to a
    hand-tuned tolerance; a 1-hour tolerance against a value stamped at midnight would match
    nothing and quietly hand you a column of NaN.

    An **instantaneous reading** such as a tide height is matched to the nearest observation,
    bounded by ``tolerance`` (defaulting to the series' own cadence), because there "how close
    is close enough" is a real judgement the caller should make.
    """
    cadence = _native_cadence(series)

    if interval:
        window = cadence
        used_direction = "backward"
        described = f"backward within its own {_pretty(window)} interval" if window else "backward"
    else:
        window = pd.Timedelta(tolerance) if tolerance is not None else cadence
        used_direction = direction
        if window is not None:
            described = f"{direction} within {_pretty(window)}"
        else:
            # One observation and no tolerance: there is no cadence to infer a sensible reach
            # from, so this matches at any distance. Say so, or a lone reading silently becomes
            # a constant column stretching across the whole query.
            described = f"{direction} (UNBOUNDED: single observation, pass tolerance= to cap)"

    # Both sides normalized to nanoseconds: a tree reopened from netCDF carries microsecond
    # datetimes while a user's timestamps are pandas-native nanoseconds, and merge_asof
    # refuses to join across resolutions rather than converting.
    left = pd.DataFrame({"__t": pd.DatetimeIndex(target).as_unit("ns")}).sort_values("__t")
    right = pd.DataFrame(
        {"__t": pd.DatetimeIndex(series.index).as_unit("ns"), "__v": series.to_numpy()}
    ).sort_values("__t")
    merged = pd.merge_asof(
        left, right, on="__t", direction=used_direction, tolerance=window
    )
    out = pd.Series(merged["__v"].to_numpy(), index=pd.DatetimeIndex(merged["__t"]))
    matched = int(out.notna().sum())
    if matched == 0:
        log.debug("no rows within %s for a series of %d points", described, len(series))
    return out, f"{described} ({matched}/{len(target)} matched)"


def _pretty(delta: pd.Timedelta | None) -> str:
    if delta is None:
        return "any gap"
    total = delta.total_seconds()
    if total % 86400 == 0:
        return f"{int(total // 86400)}d"
    if total % 3600 == 0:
        return f"{int(total // 3600)}h"
    if total % 60 == 0:
        return f"{int(total // 60)}min"
    return str(delta)


def _name_columns(wide: pd.DataFrame, style: str) -> pd.DataFrame:
    if style == "multi":
        return wide
    if style not in ("auto", "qualified"):
        raise QueryError(f"columns must be 'auto', 'qualified' or 'multi'; got {style!r}")

    counts: dict[str, int] = {}
    for variable, _station in wide.columns:
        counts[variable] = counts.get(variable, 0) + 1

    names = []
    for variable, station in wide.columns:
        if not station:  # a column carried through from the user's own frame
            names.append(variable)
        elif style == "auto" and counts[variable] == 1:
            names.append(variable)
        else:
            names.append(f"{variable}@{station}")
    out = wide.copy()
    out.columns = names
    return out


# --------------------------------------------------------------------------- redundancy


def _correlation_columns(frame: pd.DataFrame) -> list[str]:
    """The columns a linear correlation can honestly describe.

    QC flags are labels, not measurements. Compass bearings are excluded because Pearson r
    between angles is meaningless — 350 deg and 10 deg point nearly the same way and correlate
    as though they did not — which is the same reason :func:`align` combines directions as unit
    vectors instead of averaging them.
    """
    out: list[str] = []
    for column in frame.columns:
        name = str(column).lower()
        if name.endswith("_qc"):
            continue
        if "direction" in name and "spread" not in name:
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        out.append(column)
    return out


def correlations(
    frame: pd.DataFrame,
    *,
    threshold: float = 0.8,
    min_overlap: int = 10,
    method: str = "pearson",
) -> pd.DataFrame:
    """Pairs of columns that move together — the redundancy view of an aligned frame.

    One aligned frame can hold the same physical signal several times over: a station's mean,
    minimum and maximum temperature share one cadence and one week of weather; an observed
    water level and its harmonic prediction are nearly a single column; two sources five
    kilometres apart measure the same rain. A model fed all of them still predicts, but OLS
    splits the true effect arbitrarily among the near-copies and the coefficients stop meaning
    anything. This is the view that shows the problem; :func:`drop_correlated` removes it.

    Returns one row per pair with ``|r| >= threshold`` — ``feature_a``, ``feature_b``, ``r``,
    and ``n``, the overlapping samples the correlation was computed on — strongest first. Pass
    ``threshold=0`` to see every pair.

    Pairs overlapping on fewer than ``min_overlap`` samples are excluded rather than reported:
    ragged sources joined onto one axis can share only a handful of timestamps, and an r of
    1.0 over three points is noise wearing a convincing costume. QC flags and compass
    directions are excluded — see :func:`_correlation_columns` for why a linear r cannot
    describe a bearing.
    """
    columns = _correlation_columns(frame)
    if len(columns) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "r", "n"])
    numeric = frame[columns]

    corr = numeric.corr(method=method, min_periods=max(int(min_overlap), 2))
    present = numeric.notna().astype(int)
    overlap = present.T @ present

    rows: list[dict[str, Any]] = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            n = int(overlap.loc[a, b])
            r = corr.loc[a, b]
            if n < min_overlap or pd.isna(r) or abs(float(r)) < threshold:
                continue
            rows.append({"feature_a": str(a), "feature_b": str(b), "r": float(r), "n": n})

    pairs = pd.DataFrame(rows, columns=["feature_a", "feature_b", "r", "n"])
    if pairs.empty:
        return pairs
    return (
        pairs.sort_values("r", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )


def drop_correlated(
    frame: pd.DataFrame,
    *,
    threshold: float = 0.95,
    keep: Any = (),
    min_overlap: int = 10,
    method: str = "pearson",
) -> pd.DataFrame:
    """Drop one column of each highly correlated pair, and say which and why.

    Walks the pairs from :func:`correlations` strongest first; from each pair still standing it
    keeps the column with **more non-missing values** — coverage is the one merit visible in
    the frame itself — and on a tie keeps the one appearing first. Every removal is recorded in
    the result's ``attrs["omnisea_dropped"]`` as ``{dropped: reason}``, so the pruning is
    auditable rather than implicit, exactly like the resampling choices in
    ``attrs["omnisea_aggregation"]``.

    Two kinds of column are never dropped:

    * **Your own.** Columns carried through :func:`align`'s ``on=`` (recorded in
      ``attrs["omnisea_carried"]``) are the response and covariates you brought — correlation
      *with* them is the point of the model, not redundancy, so pairs touching them are
      ignored entirely rather than resolved.
    * **Pinned.** Anything named in ``keep=`` survives; its correlated partners are dropped
      instead, which is how you say "of these near-copies, this is the one I trust".

    The default threshold is deliberately conservative — 0.95 removes only near-duplicates.
    Which features belong in a model is a scientific judgment; this automates the part with a
    right answer and leaves the rest to you, with :func:`correlations` as the evidence.
    """
    pinned = {keep} if isinstance(keep, str) else {str(k) for k in keep}
    own = {str(c) for c in frame.attrs.get("omnisea_carried") or ()}

    pairs = correlations(frame, threshold=threshold, min_overlap=min_overlap, method=method)
    dropped: dict[str, str] = {}
    for row in pairs.itertuples():
        a, b = row.feature_a, row.feature_b
        if a in dropped or b in dropped or a in own or b in own:
            continue
        if a in pinned and b in pinned:
            continue
        if a in pinned:
            victim, kept = b, a
        elif b in pinned:
            victim, kept = a, b
        else:
            coverage_a = int(frame[a].notna().sum())
            coverage_b = int(frame[b].notna().sum())
            if coverage_a != coverage_b:
                victim, kept = (a, b) if coverage_a < coverage_b else (b, a)
            else:
                first_is_a = frame.columns.get_loc(a) <= frame.columns.get_loc(b)
                victim, kept = (b, a) if first_is_a else (a, b)
        dropped[victim] = f"|r|={abs(row.r):.3f} with {kept} over {row.n} samples"

    out = frame.drop(columns=list(dropped))
    out.attrs = {**frame.attrs, "omnisea_dropped": dropped}
    return out


# --------------------------------------------------------------------------- your own data


def add_local(
    tree: xr.DataTree,
    frame: pd.DataFrame,
    *,
    name: str,
    lat: float,
    lon: float,
    station_id: str | None = None,
    node_path: str = "in_situ/local",
    attrs: Mapping[str, Any] | None = None,
    var_attrs: Mapping[str, Mapping[str, Any]] | None = None,
) -> xr.DataTree:
    """Add your own observations to a tree as a node, so they travel with the rest.

    Useful when your measurements are the thing being modelled and you want one object that
    holds the response and the predictors together — it round-trips to netCDF and keeps your
    provenance beside the providers'.

    Pass ``var_attrs`` to describe your columns; a ``cell_methods`` entry there is honoured by
    :func:`align` exactly as a provider's would be.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        time_col = _find_time_column(frame)
        if time_col is None:
            raise QueryError(
                "frame needs a DatetimeIndex or a time column (time/datetime/date/timestamp)"
            )
        frame = frame.set_index(time_col)
    frame = frame.copy()
    frame.index = _to_naive_utc(pd.DatetimeIndex(frame.index))
    frame.index.name = "time"
    frame = frame.sort_index()

    ds = xr.Dataset.from_dataframe(frame)
    ds = ds.assign_coords(
        latitude=float(lat),
        longitude=float(lon),
        station_id=str(station_id or name),
        station_name=str(name),
    )
    ds["latitude"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    ds["longitude"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    ds["time"].attrs.update({"standard_name": "time", "axis": "T", "time_zone": "UTC"})
    for variable, extra in (var_attrs or {}).items():
        if variable in ds.variables:
            ds[variable].attrs.update(dict(extra))

    ds.attrs.update(
        {
            "Conventions": "CF-1.10",
            "featureType": "timeSeries",
            "provider": "local",
            "source_name": "local",
            "title": name,
            **dict(attrs or {}),
        }
    )

    path = f"/{node_path.strip('/')}/{_safe(station_id or name)}"
    contents = {
        node.path: node.dataset
        for node in tree.subtree
        if node.dataset.data_vars or node.dataset.coords
    }
    contents[path] = ds
    merged = xr.DataTree.from_dict(contents)
    merged.attrs.update(dict(tree.attrs))
    return merged


def _safe(text: Any) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(text)) or "local"
