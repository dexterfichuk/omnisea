"""What an ERDDAP dataset says about itself — the ``/info/{id}/index.json`` metadata model.

Everything the adapter knows about a dataset comes from here: the standard names, units and
cell methods are the dataset author's, not omnisea's. This module owns the parsed form of that
metadata, the vocabulary for classifying variable names (position, time, QC flag), and the
per-process cache of ``/info`` responses.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import pandas as pd

from ...errors import ProviderError
from ...query import BBox

__all__ = [
    "DEFAULT_SAMPLES_PER_DAY",
    "DatasetInfo",
    "parse_info",
    "cached_info",
    "store_info",
    "clear_cache",
    "TIME_NAMES",
    "LATITUDE_NAMES",
    "LONGITUDE_NAMES",
    "QC_STANDARD_NAMES",
    "QC_SUFFIXES",
    "is_time",
    "is_position_or_time",
    "as_float",
]

#: Assumed sampling rate when a dataset publishes no interval at all — ten minutes, a common
#: station reporting rate. It only ever feeds the Catalog estimate and the request chunking; the
#: row ceiling is enforced a second time against the rows actually returned, so a dataset that
#: samples faster than this cannot turn a bad guess into an unbounded download.
DEFAULT_SAMPLES_PER_DAY = 144.0

#: Columns tabledap returns that are position or identity rather than measurement.
TIME_NAMES = frozenset({"time"})
LATITUDE_NAMES = frozenset({"latitude", "lat"})
LONGITUDE_NAMES = frozenset({"longitude", "lon"})

#: Standard names ERDDAP/IOOS give to QC companion variables. QARTOD publishes one per test —
#: ``spike_test_quality_flag``, ``gross_range_test_quality_flag`` — so the suffix matters as much
#: as the exact names.
QC_STANDARD_NAMES = frozenset({"aggregate_quality_flag", "quality_flag", "status_flag"})
QC_SUFFIXES = ("_qc_agg", "_qc_tests", "_qartod_aggregate", "_qc", "_flag", "_flags")

#: Standard names that describe where or when a sample was taken rather than what was measured.
_COORDINATE_STANDARD_NAMES = frozenset(
    {"time", "latitude", "longitude", "depth", "altitude", "height"}
)

_info_cache: dict[tuple[str, str], DatasetInfo] = {}
_lock = threading.Lock()


def cached_info(server: str, dataset_id: str) -> DatasetInfo | None:
    """The memoized metadata for one dataset, or ``None`` if it has not been read yet."""
    with _lock:
        return _info_cache.get((server, dataset_id))


def store_info(server: str, dataset_id: str, info: DatasetInfo) -> None:
    with _lock:
        _info_cache[(server, dataset_id)] = info


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
        the first entry that is not a coordinate is the identifier. This is the one used for
        *naming*; :attr:`identity_variables` is what makes a row unique.
        """
        names = self.identity_variables
        return names[0] if names else None

    @property
    def identity_variables(self) -> tuple[str, ...]:
        """Every variable the dataset declares as identifying a series, not just the first.

        Taking only the first silently merged series that the publisher had already told us
        apart. Hakai's Pruth Bay mooring declares ``station,latitude,longitude,depth``: twelve
        depths at one station, all reporting on the same clock. Splitting on ``station`` alone
        left twelve rows competing for each timestamp, one survived, and which depth that was
        varied row to row — a "sea water temperature" series wandering between 0 m and 60 m
        with nothing to say so.
        """
        declared = self.global_attrs.get("cdm_timeseries_variables") or ""
        return tuple(
            name
            for name in (n.strip() for n in str(declared).split(","))
            if name and name in self.variables and not is_position_or_time(name)
        )

    @property
    def vertical_variable(self) -> str | None:
        """The dataset's depth/height coordinate, by its own CF declaration.

        A mooring that carries instruments at several depths reports them all on one clock, and
        the depth is what tells the readings apart. Not every dataset lists it among the
        identifying variables — DFO's ``IOS_CUR_Moorings`` declares only ``profile`` while
        publishing two instruments 80 m apart — so it is looked up directly.
        """
        for name, attrs in self.variables.items():
            if str(attrs.get("axis") or "").upper() == "Z":
                return name
        for name, attrs in self.variables.items():
            if str(attrs.get("standard_name") or "") in ("depth", "altitude", "height"):
                return name
        return None

    # ------------------------------------------------------------------ extent

    @property
    def bounds(self) -> BBox | None:
        """Published spatial extent, or ``None`` when the dataset does not declare one."""
        west = as_float(self.global_attrs.get("geospatial_lon_min"))
        east = as_float(self.global_attrs.get("geospatial_lon_max"))
        south = as_float(self.global_attrs.get("geospatial_lat_min"))
        north = as_float(self.global_attrs.get("geospatial_lat_max"))
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
            if is_position_or_time(name):
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
        if sn in QC_STANDARD_NAMES:
            return True
        return name.lower().endswith(QC_SUFFIXES)

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


# --------------------------------------------------------------------------- name vocabulary


def is_time(name: str) -> bool:
    return name in TIME_NAMES


def is_position_or_time(name: str) -> bool:
    return name in TIME_NAMES or name in LATITUDE_NAMES or name in LONGITUDE_NAMES


def _is_coordinate_or_flag_name(standard_name: str) -> bool:
    return (
        standard_name in _COORDINATE_STANDARD_NAMES
        or standard_name in QC_STANDARD_NAMES
        or standard_name.endswith("_quality_flag")
    )


# --------------------------------------------------------------------------- value helpers


def as_float(value: Any) -> float | None:
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


def _bounds_from_actual_range(variables: Mapping[str, Mapping[str, str]]) -> BBox | None:
    """Fall back to the latitude/longitude ``actual_range`` when no geospatial globals exist."""
    lat = _range_of(variables, LATITUDE_NAMES)
    lon = _range_of(variables, LONGITUDE_NAMES)
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
        parts = [as_float(p) for p in str(raw).split(",")]
        if len(parts) == 2 and None not in parts:
            return min(parts), max(parts)  # type: ignore[type-var]
    return None


def _average_spacing(dimension: str | None) -> str | None:
    """The ``averageSpacing=...`` tail of an ERDDAP dimension description."""
    if not dimension or "averageSpacing=" not in dimension:
        return None
    return dimension.split("averageSpacing=", 1)[1].strip() or None
