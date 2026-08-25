"""Template for a new omnisea data source. Copy this file and edit it.

Every hook is shown with a note on when you need it. Delete what you do not use — most sources
need only ``discover`` and ``fetch``, plus a field table.

Check your work as you go::

    python examples/provider_template.py          # runs the example below
    python -m omnisea.conformance                 # the contract, as a program

See ``docs/adding-a-provider.md`` for the full contract and ``CONTRIBUTING.md`` for the PR path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

import omnisea
from omnisea import cf
from omnisea.providers.base import (
    Provider,
    RetrievalSource,
    StationMatch,
    StationSeries,
    drop_orphan_qc,
    frame_from_records,
    trim_to_window,
)
from omnisea.query import Query


class MyOrgProvider(Provider):
    """The organization: who publishes the data, and under what terms.

    Everything here is stamped onto every node your sources produce, which is what
    ``omnisea.citation(tree)`` reads back out. Get the licence right.
    """

    name = "myorg"                       # registry key, lowercase
    title = "My Organization"            # used in attribution
    base_url = "https://api.example.org"
    license = "CC-BY-4.0"
    terms_url = "https://example.org/terms"

    def build_sources(self) -> Sequence[RetrievalSource]:
        return [MySource(self)]


class MySource(RetrievalSource):
    """One queryable dataset."""

    name = "myorg_observations"          # what users select
    title = "My Organization surface observations"
    node_path = "in_situ/myorg"          # where nodes land in the tree
    feature_type = "timeSeries"          # CF discrete sampling geometry

    # --- OPTIONAL: only set these when they apply ------------------------------------------
    #
    # period = "D"
    #     Set for rows that summarize a *period* and are labelled by its first instant (daily
    #     means, monthly totals). Without it a request starting mid-day silently drops the day
    #     it overlaps.
    #
    # retention = pd.Timedelta(days=30)
    #     Set for a rolling archive. Historical queries then get an explanation instead of an
    #     empty tree, which reads as "there is no station here".
    #
    # fields_from_metadata = True
    #     Set when the dataset publishes its own standard names and units and you read them at
    #     runtime instead of declaring the table below.
    #
    # equivalent_fields = (frozenset({"old_name", "new_name"}),)
    #     Set when two raw fields are the same measurement under different spellings and never
    #     appear in one record. Otherwise a shared variable name looks like a mistake.

    #: raw provider field -> CF description. Map what has a CF equivalent; anything you leave
    #: out still travels, under the provider's own name, tagged omnisea_mapped = 0.
    fields = {
        "water_temp_c": cf.FieldSpec(
            var="sea_water_temperature",          # variable name in the output
            standard_name="sea_water_temperature",  # MUST be in the CF table, or "" 
            units="degC",                         # units the values are ACTUALLY in
            cf_units="K",                         # optional: canonical CF units...
            cf_offset=273.15,                     # ...and how to reach them (opt-in)
            long_name="Sea water temperature",
            qc_field="water_temp_qc",             # raw field holding the flag
        ),
        "daily_rain_mm": cf.FieldSpec(
            var="precipitation_amount",
            standard_name="precipitation_amount",
            units="mm",
            # cell_methods is load-bearing: align() reads it to decide how this resamples.
            # A total must be summed, never interpolated.
            cell_methods="time: sum",
            long_name="Daily rainfall total",
        ),
    }

    # ------------------------------------------------------------------ discovery

    def discover(self, query: Query) -> list[StationMatch]:
        """What is here? Must be cheap — no bulk transfer.

        Return ``[]`` freely; a source with nothing to offer is normal, not an error.
        """
        matches: list[StationMatch] = []
        for station in self._station_list():
            if not query.contains(station["lat"], station["lon"]):
                continue
            match = self.new_match(
                station_id=station["id"],
                name=station["name"],
                lat=station["lat"],
                lon=station["lon"],
                variables=tuple(sorted(self.variables)),
                n_rows_est=int(query.days * 24),
                # Anything fetch() will need. Read it back with match.require("key") so a
                # missing value is a named error, not a silently dropped station.
                extra={"internal_id": station["id"]},
            )
            matches.append(match.attach_site(query))
        return matches

    # ------------------------------------------------------------------ retrieval

    def fetch(self, query: Query, matches: list[StationMatch]) -> list[StationSeries]:
        """Pull the confirmed subset. Let errors propagate — Catalog.fetch() decides policy."""
        out: list[StationSeries] = []
        for match in matches:
            rows = self._read_rows(match.require("internal_id"), query)
            if not rows:
                continue

            specs = cf.resolve_fields(
                self.fields,
                {key for row in rows for key in row},
                include_unmapped=self.include_unmapped(query),
                skip={"time", "station_id"},
                is_qc=lambda raw: raw.endswith("_qc"),
                # NOTE: no requested= here. `variables=` selects sources and stations; it is
                # not a projection. The response already carries every field.
            )

            to_cf = self.to_cf_units(query)
            records = []
            for row in rows:
                record: dict[str, Any] = {"time": row["time"]}
                for raw, spec in specs.items():
                    record[spec.var] = cf.convert(row.get(raw), spec, to_cf_units=to_cf)
                    if spec.qc_field and row.get(spec.qc_field) is not None:
                        record[f"{spec.var}_qc"] = row[spec.qc_field]
                records.append(record)

            frame = frame_from_records(records)
            # Enforce the window yourself: check what the upstream filter actually filters on.
            frame = trim_to_window(frame, *self.trim_window(query))
            frame = drop_orphan_qc(frame)

            out.append(
                StationSeries(
                    match=match,
                    frame=frame,
                    node_path=f"{self.node_path}/{match.station_id}",
                    attrs=self.base_attrs(
                        title=f"{match.name} ({match.station_id})",
                        source_url=f"{self.provider.base_url}/stations/{match.station_id}",
                        station_id=match.station_id,
                        site=match.site,
                    ),
                    var_attrs={
                        spec.var: cf.cf_attrs(spec, units=spec.units, to_cf_units=to_cf)
                        for spec in specs.values()
                    },
                )
            )
        return out

    # ------------------------------------------------------------------ your plumbing

    def _station_list(self) -> list[dict[str, Any]]:
        """Replace with a real call: self.provider.get_json("stations")."""
        return [{"id": "DEMO1", "name": "Demo Station", "lat": 48.8353, "lon": -125.1358}]

    def _read_rows(self, station_id: str, query: Query) -> list[dict[str, Any]]:
        """Replace with a real call. Use chunk_time() if the upstream caps its window."""
        stamps = pd.date_range(query.start, query.end, freq="h")[:24]
        return [
            {
                "time": stamp.isoformat(),
                "water_temp_c": 11.0 + index * 0.05,
                "water_temp_qc": "1",
                "daily_rain_mm": 0.0,
                "instrument_serial": "SN-4417",  # unmapped: travels under its own name
            }
            for index, stamp in enumerate(stamps)
        ]


def main() -> None:
    omnisea.register_provider(MyOrgProvider(), replace=True)

    problems = omnisea.check_source(MySource(MyOrgProvider()))
    print("conformance:", "clean" if not problems else "\n".join(map(str, problems)))

    tree = omnisea.fetch(
        lat=48.8353, lon=-125.1358, radius_km=10,
        time=("2024-07-01", "2024-07-02"), providers="myorg",
    )
    print(omnisea.summary(tree)[["node", "n_time", "variables"]].to_string(index=False))
    print()
    print(omnisea.citation(tree))


if __name__ == "__main__":
    main()
