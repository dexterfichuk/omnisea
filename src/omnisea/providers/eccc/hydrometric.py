"""ECCC hydrometric: water level and river discharge."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ... import cf
from ...query import Query
from ..base import StationMatch
from ..ogc import OgcFeaturesSource

__all__ = ["EcccHydrometric"]


class EcccHydrometric(OgcFeaturesSource):
    """Realtime water level and discharge from the national hydrometric network."""

    name = "eccc_hydrometric"
    title = "ECCC hydrometric realtime"
    node_path = "in_situ/hydrometric"
    collection = "hydrometric-realtime"
    station_collection = "hydrometric-stations"
    station_id_field = "STATION_NUMBER"
    catalogue_id_field = "STATION_NUMBER"
    time_field = "DATETIME"
    skip_fields = frozenset(
        {
            "IDENTIFIER",
            "STATION_NUMBER",
            "STATION_NAME",
            "PROV_TERR_STATE_LOC",
            "DATETIME",
            "DATETIME_LST",
        }
    )
    qc_suffix = ""
    samples_per_day = 24.0 * 4  # 15-minute reporting is typical

    fields = {
        "LEVEL": cf.FieldSpec(
            var="water_surface_height_above_reference_datum",
            standard_name="water_surface_height_above_reference_datum",
            units="m", long_name="Water level", qc_field="LEVEL_SYMBOL_EN",
        ),
        "DISCHARGE": cf.FieldSpec(
            var="water_volume_transport_in_river_channel",
            standard_name="water_volume_transport_in_river_channel",
            units="m3 s-1", long_name="River discharge", qc_field="DISCHARGE_SYMBOL_EN",
        ),
    }

    def is_qc_field(self, raw: str) -> bool:
        return raw.endswith(("_SYMBOL_EN", "_SYMBOL_FR"))

    def qc_field_for(self, raw: str, spec: cf.FieldSpec) -> str | None:
        return spec.qc_field

    def station_from_feature(
        self, query: Query, feature: Mapping[str, Any]
    ) -> StationMatch | None:
        match = super().station_from_feature(query, feature)
        if match is None:
            return None
        props = feature.get("properties") or {}
        # The catalogue lists discontinued gauges too; realtime data only exists for active ones.
        if props.get("REAL_TIME") in (0, "0", False):
            return None
        match.extra["vertical_datum"] = props.get("VERTICAL_DATUM") or None
        match.extra["status"] = props.get("STATUS_EN")
        return match

    def node_attrs(self, query: Query, match: StationMatch) -> dict[str, Any]:
        attrs = super().node_attrs(query, match)
        datum = match.extra.get("vertical_datum")
        if datum:
            attrs["datum"] = datum
        return attrs
