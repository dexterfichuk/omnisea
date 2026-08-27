"""A complete third-party omnisea provider, in one file.

Reads station observations from a directory of CSVs — no network, no credentials — so it can be
run and tested anywhere. It exists to be copied: the structure here is the same structure the
built-in providers use, minus the HTTP.

Layout it expects::

    data/
      stations.csv     id,name,lat,lon
      BAM01.csv        time,water_temp_c,qc,battery_v
      BAM02.csv        ...

Run it::

    python examples/csv_stations.py

See ``docs/adding-a-provider.md`` for the full interface contract.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import omnisea
from omnisea import cf
from omnisea.providers.base import (
    Provider,
    RetrievalSource,
    StationSeries,
    drop_orphan_qc,
    frame_from_records,
    trim_to_window,
)
from omnisea.query import Query


class ShoreLoggerProvider(Provider):
    """The organization: who publishes the data, and under what licence."""

    name = "shorelogger"
    title = "Shore Logger Network"
    base_url = "file://./data"
    license = "CC-BY-4.0"
    terms_url = "https://creativecommons.org/licenses/by/4.0/"

    def __init__(self, root: str | Path = "data") -> None:
        super().__init__()
        self.root = Path(root)

    def build_sources(self) -> Sequence[RetrievalSource]:
        return [ShoreLoggerSource(self)]

    def station_list(self) -> list[dict[str, Any]]:
        index = self.root / "stations.csv"
        if not index.exists():
            return []
        with index.open(newline="", encoding="utf-8") as fh:
            return [
                {
                    "id": row["id"],
                    "name": row.get("name", row["id"]),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                }
                for row in csv.DictReader(fh)
            ]

    def read_rows(self, station_id: str) -> list[dict[str, Any]]:
        path = self.root / f"{station_id}.csv"
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))


class ShoreLoggerSource(RetrievalSource):
    """The dataset: what the fields mean and how to get them."""

    name = "shorelogger_sst"
    title = "Shore logger sea surface temperature"
    node_path = "in_situ/shore_logger"
    feature_type = "timeSeries"

    #: Only `water_temp_c` has a CF equivalent. `battery_v` is carried through unmapped rather
    #: than dropped — omnisea canonicalizes what it can and keeps what it cannot.
    fields = {
        "water_temp_c": cf.FieldSpec(
            var="sea_water_temperature",
            standard_name="sea_water_temperature",
            units="degC",
            cf_units="K",
            cf_offset=273.15,
            long_name="Sea water temperature",
            qc_field="qc",
        ),
    }

    def discover(self, query: Query) -> list[omnisea.StationMatch]:
        matches = []
        for station in self.provider.station_list():
            if not query.contains(station["lat"], station["lon"]):
                continue
            match = self.new_match(
                station_id=station["id"],
                name=station["name"],
                lat=station["lat"],
                lon=station["lon"],
                variables=tuple(sorted(self.variables)),
                n_rows_est=int(query.days * 24),
            )
            matches.append(match.attach_site(query))
        return matches

    def fetch(self, query: Query, matches: list[omnisea.StationMatch]) -> list[StationSeries]:
        out: list[StationSeries] = []
        for match in matches:
            rows = self.provider.read_rows(match.station_id)
            if not rows:
                continue

            specs = cf.resolve_fields(
                self.fields,
                {key for row in rows for key in row},
                requested=query.variables,
                include_unmapped=self.include_unmapped(query),
                skip={"time", "station_id"},
                is_qc=lambda raw: raw == "qc",
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

            frame = trim_to_window(frame_from_records(records), query.start, query.end)
            out.append(
                StationSeries(
                    match=match,
                    frame=drop_orphan_qc(frame),
                    node_path=f"{self.node_path}/{match.station_id}",
                    attrs=self.base_attrs(
                        title=f"{match.name} ({match.station_id})",
                        station_id=match.station_id,
                        site=match.site,
                    ),
                    var_attrs={
                        spec.var: cf.cf_attrs(spec, to_cf_units=to_cf)
                        for spec in specs.values()
                    },
                )
            )
        return out


def main() -> None:
    import tempfile

    # Build a tiny dataset so the example runs standalone.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "stations.csv").write_text(
            "id,name,lat,lon\nBAM01,Bamfield Inlet Logger,48.8353,-125.1358\n", encoding="utf-8"
        )
        (root / "BAM01.csv").write_text(
            "time,water_temp_c,qc,battery_v\n"
            "2024-07-01T00:00:00Z,11.4,1,12.8\n"
            "2024-07-01T01:00:00Z,11.6,1,12.8\n"
            "2024-07-01T02:00:00Z,11.9,1,12.7\n",
            encoding="utf-8",
        )

        omnisea.register_provider(ShoreLoggerProvider(root), replace=True)
        print("registered sources:", omnisea.sources())

        tree = omnisea.fetch(
            lat=48.8353,
            lon=-125.1358,
            radius_km=5,
            time=("2024-07-01", "2024-07-02"),
            providers="shorelogger",
        )
        print(tree)
        print("\n", omnisea.summary(tree).to_string(index=False))


if __name__ == "__main__":
    main()
