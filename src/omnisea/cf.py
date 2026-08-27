"""Semantic canonicalization: provider field names -> CF standard names.

Each provider declares a table of :class:`FieldSpec`, and everything downstream — variable
naming, units, QC columns, the ``variables()`` listing — is derived from those tables.

Two rules govern the numbers themselves:

* **Encoding fixes are always applied.** ECCC ships wind direction in *tens of degrees*, so a
  raw ``25`` means 250 degrees. That is a storage encoding, not a unit choice, and leaving it
  raw would be simply wrong.
* **Unit conversions are opt-in.** Values stay in the units the provider published, with those
  units recorded in the ``units`` attribute. Pass ``to_cf_units=True`` to convert to canonical
  CF units. Silently turning 15 degC into 288.15 K would make a scientist distrust their own
  data at first glance.

Every ``standard_name`` here is checked against the CF standard name table; note that
``sea_surface_height_above_reference_datum`` is *not* a CF standard name (the real one is
``water_surface_height_above_reference_datum``), so that spelling is accepted as an input alias
but never emitted.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Container, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = [
    "FieldSpec",
    "convert",
    "cf_attrs",
    "resolve_fields",
    "passthrough_spec",
    "resolve_names",
    "EOV_TO_CF",
    "eov_to_cf",
]


@dataclass(frozen=True)
class FieldSpec:
    """How one raw provider field becomes one CF-described variable.

    ``scale``/``offset`` repair the provider's encoding and are always applied.
    ``cf_scale``/``cf_offset`` convert the repaired value into ``cf_units`` and apply only when
    the caller asks for canonical units.
    """

    var: str
    standard_name: str = ""  # empty when the quantity has no CF standard name (e.g. humidex)
    units: str | None = None  # None -> the provider supplies units per-record (e.g. SWOB `-uom`)
    cf_units: str | None = None
    scale: float = 1.0
    offset: float = 0.0
    cf_scale: float = 1.0
    cf_offset: float = 0.0
    cell_methods: str | None = None
    long_name: str | None = None
    qc_field: str | None = None
    comment: str | None = None
    extra_attrs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def canonical_units(self) -> str | None:
        return self.cf_units or self.units


# --------------------------------------------------------------------------- name resolution

#: Names people reasonably reach for that are not the CF spelling omnisea emits.
#: ``sea_surface_height_above_reference_datum`` in particular looks official but is absent from
#: the CF standard name table, so it is accepted on input and mapped to the real name.
ALIASES: dict[str, str] = {
    "sea_surface_height_above_reference_datum": "water_surface_height_above_reference_datum",
    "sea_surface_height": "water_surface_height_above_reference_datum",
    "water_level": "water_surface_height_above_reference_datum",
    "tide": "water_surface_height_above_reference_datum",
    "temperature": "air_temperature",
    "temp": "air_temperature",
    "pressure": "air_pressure",
    "humidity": "relative_humidity",
    "precipitation": "precipitation_amount",
    "discharge": "water_volume_transport_in_river_channel",
    "streamflow": "water_volume_transport_in_river_channel",
}


def resolve_names(names: Iterable[str] | None) -> frozenset[str] | None:
    """Expand user-supplied variable names through :data:`ALIASES`.

    Both the alias and its target are kept, so asking for ``air_temperature`` still matches the
    ``air_temperature_max`` variable via its ``standard_name``.
    """
    if names is None:
        return None
    out: set[str] = set()
    for n in names:
        key = str(n).strip()
        out.add(key)
        if key in ALIASES:
            out.add(ALIASES[key])
    return frozenset(out) if out else None


# --------------------------------------------------------------------------- passthrough

#: Marker attribute distinguishing a field omnisea understands from one it merely carries.
MAPPED_ATTR = "omnisea_mapped"


def _safe_var_name(raw: str) -> str:
    """A netCDF-legal variable name that still reads as the provider's own field name."""
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(raw))
    # Collapse runs, so "avg wind-speed (10m)" reads as avg_wind_speed_10m, not ..._10m.
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return "unnamed"
    if cleaned[0].isdigit():
        cleaned = f"v_{cleaned}"
    return cleaned


def passthrough_spec(raw: str, *, units: str | None = None) -> FieldSpec:
    """Carry an unmapped provider field through under its own name.

    omnisea canonicalizes what it can and *keeps* what it cannot. A field with no CF standard
    name still travels, under the provider's own spelling, tagged so you can tell at a glance
    which variables are canonical and which are raw:

        ds["TEMP"]                 # -> air_temperature, standard_name set
        ds["WEATHER_ENG_DESC"]     # -> carried verbatim, omnisea_mapped = 0

    Dropping these would make omnisea lossy, which defeats the point of a unified client.
    """
    return FieldSpec(
        var=_safe_var_name(raw),
        standard_name="",
        units=units,
        long_name=str(raw),
        extra_attrs={MAPPED_ATTR: 0, "source_field": str(raw)},
    )


def _spec_matches(spec: FieldSpec, raw: str, wanted: frozenset[str]) -> bool:
    """Does this field answer the caller's variable request?

    Matching accepts the CF standard name, omnisea's variable name, or the provider's own raw
    field name, so ``variables=["TEMP"]`` works as naturally as
    ``variables=["air_temperature"]``.
    """
    return bool(
        spec.var in wanted
        or (spec.standard_name and spec.standard_name in wanted)
        or raw in wanted
    )


def resolve_fields(
    table: Mapping[str, FieldSpec],
    available: Iterable[str],
    *,
    requested: Iterable[str] | None = None,
    include_unmapped: bool = True,
    skip: Container[str] = frozenset(),
    is_qc: Callable[[str], bool] | None = None,
    units_for: Callable[[str], str | None] | None = None,
) -> dict[str, FieldSpec]:
    """Decide which raw fields to emit, and how to describe each one.

    ``table`` is the source's own field mapping. ``available`` is what the response actually
    contained, so passthrough variables reflect real data rather than a guessed schema.
    Identity/time columns go in ``skip``; QC siblings are recognised by ``is_qc`` and attached
    to their parent variable instead of becoming variables of their own.

    ``requested`` narrows the emitted columns. The built-in sources deliberately do **not** use
    it: their responses already carry every property, so dropping columns would discard data
    that has already crossed the network. It exists for sources whose upstream bills per field
    or requires naming them in the request.
    """
    wanted = resolve_names(requested)
    out: dict[str, FieldSpec] = {}
    taken: set[str] = set()

    for raw in available:
        if raw in skip:
            continue
        if is_qc is not None and is_qc(raw) and raw not in table:
            continue

        spec = table.get(raw)
        if spec is None:
            if not include_unmapped:
                continue
            spec = passthrough_spec(raw, units=units_for(raw) if units_for else None)
        elif units_for is not None and spec.units is None:
            # Providers that publish units per record (SWOB) fill them in here.
            resolved = units_for(raw)
            if resolved:
                spec = replace(spec, units=resolved)

        if wanted is not None and not _spec_matches(spec, raw, wanted):
            continue

        # Two raw fields must never collapse onto one variable name.
        name = spec.var
        if name in taken:
            suffix = 2
            while f"{name}_{suffix}" in taken:
                suffix += 1
            name = f"{name}_{suffix}"
            spec = replace(spec, var=name)
        taken.add(name)
        out[raw] = spec

    return out


# --------------------------------------------------------------------------- value + attrs

def convert(value: Any, spec: FieldSpec, *, to_cf_units: bool = False) -> Any:
    """Apply the encoding fix, then optionally the unit conversion.

    ``None`` passes straight through so that missing values stay missing rather than becoming
    ``273.15``.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    v = v * spec.scale + spec.offset
    if to_cf_units:
        v = v * spec.cf_scale + spec.cf_offset
    return v


def cf_attrs(
    spec: FieldSpec, *, units: str | None = None, to_cf_units: bool = False
) -> dict[str, Any]:
    """CF attributes for one variable.

    ``units`` overrides the table for providers that publish units per record (SWOB). Whichever
    units the values are actually in is what lands in the ``units`` attribute — that invariant is
    what makes the opt-in conversion safe.
    """
    resolved_units = units if units is not None else spec.units
    if to_cf_units and spec.cf_units:
        emitted_units = spec.cf_units
    else:
        emitted_units = resolved_units

    attrs: dict[str, Any] = {}
    if spec.standard_name:
        attrs["standard_name"] = spec.standard_name
    if emitted_units:
        attrs["units"] = emitted_units
    if to_cf_units and spec.cf_units and spec.cf_units != resolved_units:
        # Say so on the variable itself. Before this, a converted value carried only its new
        # units and dropped both `cf_units` and the note — so a reader of the netCDF could not
        # tell a value omnisea had converted from one the provider published that way. In a
        # library that records every resampling choice, that was the one transformation with
        # no trace.
        attrs["omnisea_converted_from"] = resolved_units or "(unstated)"
        attrs["comment_units"] = (
            f"converted by omnisea from {resolved_units} to {spec.cf_units} "
            "(to_cf_units=True); the provider published the former"
        )
    if spec.long_name:
        attrs["long_name"] = spec.long_name
    if spec.cell_methods:
        attrs["cell_methods"] = spec.cell_methods
    if spec.comment:
        attrs["comment"] = spec.comment
    if not to_cf_units and spec.cf_units and spec.cf_units != resolved_units:
        attrs["cf_units"] = spec.cf_units
        attrs["note"] = (
            f"values are in provider units ({resolved_units}); "
            f"pass to_cf_units=True for {spec.cf_units}"
        )
    if spec.standard_name:
        attrs.setdefault(MAPPED_ATTR, 1)
    attrs.update(spec.extra_attrs)
    return attrs


# --------------------------------------------------------------------------- GOOS EOV bridge

#: GOOS Essential Ocean Variables (the vocabulary CIOOS metadata records use) mapped to the CF
#: standard names omnisea speaks. Biological and ecosystem EOVs have no single CF equivalent and
#: are deliberately absent rather than forced into an approximate match.
EOV_TO_CF: dict[str, tuple[str, ...]] = {
    "seaSurfaceHeight": ("water_surface_height_above_reference_datum",),
    "seaSurfaceTemperature": ("sea_surface_temperature",),
    "subSurfaceTemperature": ("sea_water_temperature",),
    "seaSurfaceSalinity": ("sea_surface_salinity",),
    "subSurfaceSalinity": ("sea_water_practical_salinity",),
    "surfaceCurrents": (
        "surface_eastward_sea_water_velocity",
        "surface_northward_sea_water_velocity",
    ),
    "subSurfaceCurrents": ("eastward_sea_water_velocity", "northward_sea_water_velocity"),
    "oxygen": ("mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",),
    "nutrients": ("mole_concentration_of_nitrate_in_sea_water",),
    "inorganicCarbon": ("mole_concentration_of_dissolved_inorganic_carbon_in_sea_water",),
    "dissolvedOrganicCarbon": ("mole_concentration_of_dissolved_organic_carbon_in_sea_water",),
    "seaIce": ("sea_ice_area_fraction",),
    "seaState": ("sea_surface_wave_significant_height",),
    "oceanColour": ("mass_concentration_of_chlorophyll_in_sea_water",),
    "oceanSurfaceHeatFlux": ("surface_downward_heat_flux_in_sea_water",),
    "oceanSurfaceStress": ("surface_downward_eastward_stress",),
    "particulateMatter": ("mass_concentration_of_suspended_matter_in_sea_water",),
}

#: Reverse index, so a CF-name query can be matched against EOV-tagged metadata records.
CF_TO_EOV: dict[str, tuple[str, ...]] = {}
for _eov, _cfs in EOV_TO_CF.items():
    for _cf in _cfs:
        CF_TO_EOV[_cf] = CF_TO_EOV.get(_cf, ()) + (_eov,)


def eov_to_cf(eovs: Iterable[str] | None) -> tuple[str, ...]:
    """CF standard names implied by a record's EOV tags (unmapped EOVs are dropped)."""
    if not eovs:
        return ()
    out: list[str] = []
    for e in eovs:
        for cf in EOV_TO_CF.get(str(e), ()):
            if cf not in out:
                out.append(cf)
    return tuple(out)
