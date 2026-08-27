"""``tabledap`` — station, mooring, profile and trajectory records, as point time series.

The field table is built per dataset from its own ``/info`` metadata (:func:`field_table`), so
values come back CF-described exactly as their author described them, and a dataset holding many
platforms is split into one series per station rather than collapsed onto one time axis.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from ... import cf
from ...errors import PayloadTooLargeError
from ...http import DEFAULT_MAX_WORKERS, chunk_time, map_threads
from ...query import Query
from ..base import (
    StationMatch,
    StationSeries,
    drop_orphan_qc,
    frame_from_records,
    trim_to_window,
)
from .common import ErddapSource, safe_name, table_rows
from .info import (
    LATITUDE_NAMES,
    LONGITUDE_NAMES,
    TIME_NAMES,
    DatasetInfo,
    as_float,
    is_time,
)

log = logging.getLogger("omnisea.erddap")

__all__ = ["ErddapTableSource", "field_table", "ROWS_PER_REQUEST"]

#: Rows to aim for in one tabledap request. Long windows are split to this size rather than asked
#: for in one go, because ERDDAP builds the whole response in memory before sending it.
ROWS_PER_REQUEST = 100_000

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


class ErddapTableSource(ErddapSource):
    """``tabledap`` — station, mooring, profile and trajectory records, as point time series."""

    name = "erddap_tabledap"
    title = "ERDDAP tabledap"
    node_path = "in_situ/erddap"
    feature_type = "timeSeries"
    protocol = "tabledap"
    data_structure = "table"

    def unusable_reason(self, info: DatasetInfo) -> str | None:
        """Refuse a table with no ``time`` variable, and say what it has instead.

        Not every tabledap dataset is a time series. CIOOS Pacific's ``IOS_P26_Annualized``
        (Ocean Station Papa) is indexed by an integer ``Year`` column and publishes no ``time``
        at all, so the ``&time>=`` constraint every request carries comes back as
        ``400 Unrecognized constraint variable="time"`` — an upstream error the user can do
        nothing with. Turning a year column into a time axis would mean inventing an instant
        for each row, the same reason ``climate-normals`` is unsupported rather than guessed at.
        """
        if "time" in info.variables:
            return None
        candidates = [
            name for name in info.variables
            if any(word in name.lower() for word in ("year", "date", "month", "day"))
        ]
        hint = f" It is indexed by {', '.join(candidates)}." if candidates else ""
        return (
            "the dataset publishes no 'time' variable, so it has no time axis to place rows "
            f"on.{hint} omnisea will not invent one"
        )

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
        series = self.series_from_rows(query, match, info, rows) if rows else []
        if not series:
            # Name it rather than returning nothing. A dataset can match on metadata and still
            # hold no usable rows — CIOOS Pacific's IYS_2019_CTD declares a `time` variable
            # that is entirely empty, so ERDDAP answers every window with "no matching
            # results" — and a tree that simply lacks the node reads as "there is nothing
            # here". An empty series is recorded in the tree's omnisea_empty_stations and
            # reported by citation(), which is what the OGC sources have always done.
            return [
                StationSeries(
                    match=match,
                    frame=pd.DataFrame(),
                    node_path=f"{self.node_path}/{safe_name(info.dataset_id)}",
                    attrs=self._node_attrs(info, server),
                )
            ]
        return series

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
            rows.extend(table_rows(payload))
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
        include_unmapped = self.include_unmapped(query)
        table = field_table(
            info, present=_ordered_keys(rows), include_unmapped=include_unmapped
        )
        skip = _skip_columns(info, station_var)
        primary_qc = _primary_qc_names(info)
        to_cf = self.to_cf_units(query)

        out: list[StationSeries] = []
        for station_id, group in groups.items():
            specs = cf.resolve_fields(
                table,
                _ordered_keys(group),
                include_unmapped=include_unmapped,
                skip=skip,
                is_qc=lambda raw: raw in primary_qc,
                units_for=lambda raw: _units_of(info, raw),
            )
            frame, var_attrs = self._frame_for(query, group, specs, to_cf)
            if frame.empty:
                continue

            member = self._member_match(query, match, info, station_id, group)
            path = f"{self.node_path}/{safe_name(info.dataset_id)}"
            if station_id is not None and len(groups) > 1:
                path = f"{path}/{safe_name(station_id)}"
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
        query: Query,
        match: StationMatch,
        info: DatasetInfo,
        station_id: Any,
        rows: list[Mapping[str, Any]],
    ) -> StationMatch:
        """The match this series belongs to: the dataset's, or a per-station copy of it.

        Position comes from the rows rather than from the dataset's declared bounding box. For a
        fixed station the two are the same; for a dataset holding many platforms the box is the
        envelope of all of them, and for one that declared no extent at all it is the only
        position there is. The distance to the requested site is recomputed from it, so the
        position a Catalog row shows and the distance beside it always describe the same point.
        """
        lat, lon = _mean_position(rows)
        if station_id is None and (lat is None or lon is None):
            return match

        named = station_id is not None
        member = self.new_match(
            station_id=f"{info.dataset_id}:{station_id}" if named else match.station_id,
            name=f"{info.title} — {station_id}" if named else match.name,
            lat=lat if lat is not None else match.lat,
            lon=lon if lon is not None else match.lon,
            variables=match.variables,
            n_rows_est=len(rows) if named else match.n_rows_est,
            first=match.first,
            last=match.last,
            extra={**match.extra, "erddap_station": station_id} if named else dict(match.extra),
        )
        return member.attach_site(query)


# --------------------------------------------------------------------------- field tables


def field_table(
    info: DatasetInfo, *, present: Iterable[str], include_unmapped: bool = True
) -> dict[str, cf.FieldSpec]:
    """Build the CF field table for one dataset from its own published metadata.

    The output variable takes the CF standard name where the dataset gives one and that name is
    unambiguous within the dataset; where two variables share a standard name — the same quantity
    at two depths, say — both keep their ERDDAP names, because a ``_2`` suffix would tell a
    reader nothing about which is which.

    A variable with no standard name is described here too rather than left to
    :func:`cf.passthrough_spec`, because the dataset still published a ``long_name``, units and
    possibly ``cell_methods`` for it, and the generic passthrough would throw all three away. It
    is tagged ``omnisea_mapped = 0`` just the same, and dropped when the caller does not want
    unmapped fields.
    """
    present = list(present)
    qc_map = info.qc_map()
    primary_qc = _primary_qc_names(info)
    all_qc = {name for names in qc_map.values() for name in names}

    emitted = [
        name
        for name in present
        if name in info.variables and name not in all_qc and not is_time(name)
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
        if not standard_name and not include_unmapped:
            continue
        var = standard_name if (standard_name and counts.get(standard_name, 0) == 1) else name
        if name in _POSITION_RENAME:
            var = _POSITION_RENAME[name]
        qc_field = next((q for q in qc_map.get(name, ()) if q in primary_qc), None)

        extra: dict[str, Any] = {"source_field": name}
        if not standard_name:
            extra[cf.MAPPED_ATTR] = 0
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
    skip = set(TIME_NAMES)
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
        lat_name = _named_variable(info, LATITUDE_NAMES)
        lon_name = _named_variable(info, LONGITUDE_NAMES)
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
    lats = [as_float(_first_of(row, LATITUDE_NAMES)) for row in rows]
    lons = [as_float(_first_of(row, LONGITUDE_NAMES)) for row in rows]
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
