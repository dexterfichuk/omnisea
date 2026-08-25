# Adding a provider

omnisea is designed so that a new data source is **one new file**, not a refactor. This document
is the contract: implement it and your source is discovered, filtered, fetched, CF-described and
assembled into the tree alongside every built-in source, with no changes to omnisea itself.

---

## The two levels

| Level | Class | What it is | Example |
|---|---|---|---|
| Organization | `Provider` | Who publishes the data. Owns base URL, licence, attribution, auth. | `eccc` |
| Dataset | `DataSource` | One queryable thing. Owns the field table, node path, discover/fetch. | `eccc_climate` |

One provider may publish many sources. ECCC publishes four. Users select either — naming a
provider selects all of its sources:

```python
omnisea.fetch(..., providers="eccc")          # all four ECCC datasets
omnisea.fetch(..., providers="eccc_climate")  # just hourly climate
```

Pick the right base class for your dataset:

| Base class | Use when | Must implement |
|---|---|---|
| `RetrievalSource` | Your source returns data arrays. | `discover()`, `fetch()` |
| `DiscoverySource` | Your source describes data and hands back URLs (metadata catalogues, STAC). | `discover()` |
| `OgcFeaturesSource` | Your source is an OGC API - Features / pygeoapi collection. | usually just class attributes |

---

## The contract

### `Provider`

```python
class Provider(ABC):
    name: str        # registry key for the organization, e.g. "noaa"
    title: str       # human-readable, used in attribution
    base_url: str    # root URL its sources hang off
    license: str     # recorded on every node this provider produces
    terms_url: str   # where the licence lives

    def build_sources(self) -> Sequence[DataSource]: ...
```

`build_sources()` is called once, lazily. Return instantiated sources, passing `self`.

Free for you: `self.get_json(path, params)` (retries, backoff, User-Agent, shared connection
pool, global concurrency cap) and `self.attribution()`.

### `DataSource`

```python
class DataSource(ABC):
    name: str                       # registry key, e.g. "noaa_coops"
    title: str                      # human-readable dataset name
    node_path: str                  # where nodes live, e.g. "in_situ/tides"
    feature_type: str = "timeSeries"  # CF DSG type
    fields: dict[str, FieldSpec]    # raw field name -> CF description

    def discover(self, query: Query) -> list[StationMatch]: ...
    def fetch(self, query, matches) -> list[StationSeries | xr.Dataset]: ...
```

**`discover(query)` must be cheap.** It answers "what is here, and roughly how much?" so a user
can look before downloading. Do not pull observations here. Return `[]` freely — a source with
nothing to offer for this query is normal, not an error.

**Passing state from `discover()` to `fetch()`** goes through `StationMatch.extra`, an
untyped dict for whatever your fetch step needs — an internal id, a series code — that the
catalogue itself should not display. Read required keys with `match.require("iwls_id")` rather
than `match.extra.get(...)`: a missing key then raises a named error instead of quietly
dropping the station, and for scientific data that is the difference between a bug you find and
one you publish.

**`fetch(query, matches)`** receives only the matches that survived the user's filtering. Return
either:

- `StationSeries` — a time-indexed `DataFrame` plus attributes. omnisea builds the CF dataset,
  the coordinates and the tree node for you. This is the point path.
- `xr.Dataset` — a ready-made dataset, for gridded sources. Set
  `attrs["omnisea_node_path"]` to place it. Keep it lazy (Dask); do not call `.load()`.

That union is the seam that lets a gridded source (Copernicus, ERDDAP griddap) drop in without
touching the point-series assembly code.

### Helpers you get for free

| Helper | What it does |
|---|---|
| `self.new_match(...)` | A `StationMatch` pre-tagged with your source and provider |
| `self.base_attrs(**extra)` | Node attrs with `Conventions`, `featureType`, licence, institution |
| `self.include_unmapped(query)` | Whether to carry fields with no CF mapping (default yes) |
| `self.to_cf_units(query)` | Whether the caller asked for canonical CF units |
| `self.covers(query)` / `self.retention_gap(query)` | Whether your rolling window reaches the requested dates, and the message if not |
| `match.attach_site(query)` | Records which requested site a station answers for, and how far |
| `match.require(key)` | Read an `extra` value `discover()` promised `fetch()`, failing loudly if absent |
| `frame_from_records(rows)` | Time-indexed, sorted, de-duplicated frame from row dicts |
| `trim_to_window(frame, start, end)` | Enforce the requested window regardless of upstream filter semantics |
| `drop_orphan_qc(frame)` | Drop `<var>_qc` whose measurement column was empty |
| `cf.convert(value, spec)` | Apply the encoding fix (and optionally the unit conversion) |
| `cf.cf_attrs(spec)` | CF attributes for one variable |
| `cf.resolve_fields(...)` | Decide which raw fields to emit, mapped and unmapped |
| `chunk_time(start, end, max_days=N)` | Split a window to respect an upstream interval cap |
| `map_threads(fn, items)` | Bounded parallelism (HTTP concurrency is capped globally anyway) |

---

## Describing your variables: `FieldSpec`

`FieldSpec` maps one raw provider field to one CF-described variable.

```python
cf.FieldSpec(
    var="air_temperature",              # variable name in the output Dataset
    standard_name="air_temperature",    # CF standard name; "" if none exists
    units="degC",                       # units as the provider publishes them
    cf_units="K", cf_offset=273.15,     # how to reach canonical CF units (opt-in)
    scale=1.0, offset=0.0,              # encoding repair, ALWAYS applied
    cell_methods="time: mean",
    long_name="Air temperature",
    qc_field="TEMP_FLAG",               # raw field holding the QC flag
    comment="...",
)
```

Two rules, and the distinction between them matters:

- **`scale`/`offset` repair the provider's encoding and are always applied.** ECCC ships wind
  direction in tens of degrees, so raw `25` means 250°. That is a storage encoding, not a unit
  choice — leaving it raw would be simply wrong.
- **`cf_scale`/`cf_offset` convert units and apply only when the caller passes
  `to_cf_units=True`.** Values otherwise stay in the units the provider published, with those
  units in the `units` attribute. Silently turning 15 °C into 288.15 K makes a scientist
  distrust their own data at a glance.

**Declare `retention` if your dataset is a rolling window.** A source that keeps only the last
30 days should say so:

```python
class MyRealtimeSource(RetrievalSource):
    retention = pd.Timedelta(days=30)
```

omnisea then checks the query window *before* calling you, skips the request when it cannot
possibly help, and prints an explanation on the Catalog. Without it, a historical query gets an
empty result — which a user reads as "there is no station here", a different and wrong
conclusion from "this collection only keeps 30 days". Leave it `None` for a full archive.

**Set `cell_methods` whenever a value summarizes an interval** — a total, a mean, a maximum
over a period. It is not decoration: `omnisea.align()` reads it to decide how the variable
resamples and how it joins to a user's own timestamps. A daily total without `cell_methods`
gets interpolated as if it were an instantaneous reading, which invents an intra-day
distribution that was never measured.

**Verify your `standard_name` against the real CF table.** `sea_surface_height_above_reference_datum`
looks official and is not in it; the real name is `water_surface_height_above_reference_datum`.
A CI test checks every built-in name against the published table — yours should pass it too.
If no CF standard name exists for a quantity (humidex, battery voltage), set `standard_name=""`
and give it a good `long_name`. Do not invent one.

**You do not need to map everything — and you should not filter.** Fields absent from `fields`
are carried through under their own provider names, tagged `omnisea_mapped = 0`. Map what has a
CF equivalent; the rest still travels.

Do **not** pass `requested=query.variables` to `cf.resolve_fields` if your response already
contains every field. `variables=` selects which sources and stations to fetch; it is not a
projection over what they return, and dropping columns you already downloaded is pure loss. The
parameter exists only for upstreams that bill per field or require naming them in the request.

The same reasoning governs `wants_anything()`, which you inherit: a curated table is a floor,
not an inventory. If a caller asks for a name you do not recognise, the base class keeps your
source in play rather than opting out, because the field may be one your platform publishes and
omnisea has no CF name for.

---

## A complete example

`examples/csv_stations.py` is a working third-party provider that reads station CSVs from a
local directory. It is under 100 lines and is exercised by the test suite.

```python
import omnisea
from omnisea.providers.base import Provider, RetrievalSource, StationSeries, frame_from_records

class ShoreLoggerProvider(Provider):
    name = "shorelogger"
    title = "Shore Logger Network"
    base_url = "https://example.org/shorelogger"
    license = "CC-BY-4.0"

    def build_sources(self):
        return [ShoreLoggerSource(self)]


class ShoreLoggerSource(RetrievalSource):
    name = "shorelogger_sst"
    title = "Shore logger sea surface temperature"
    node_path = "in_situ/shore_logger"
    feature_type = "timeSeries"

    fields = {
        "water_temp_c": cf.FieldSpec(
            var="sea_water_temperature",
            standard_name="sea_water_temperature",
            units="degC", cf_units="K", cf_offset=273.15,
            long_name="Sea water temperature",
            qc_field="qc",
        ),
    }

    def discover(self, query):
        matches = []
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
            )
            matches.append(match.attach_site(query))
        return matches

    def fetch(self, query, matches):
        out = []
        for match in matches:
            rows = self._read_rows(match.station_id)
            specs = cf.resolve_fields(
                self.fields, {k for r in rows for k in r},
                requested=query.variables,
                include_unmapped=self.include_unmapped(query),
                skip={"time", "station_id"},
            )
            to_cf = self.to_cf_units(query)   # honour the caller's units choice
            records = [
                {"time": r["time"], **{s.var: cf.convert(r.get(raw), s, to_cf_units=to_cf)
                                       for raw, s in specs.items()}}
                for r in rows
            ]
            frame = trim_to_window(frame_from_records(records), query.start, query.end)
            out.append(StationSeries(
                match=match,
                frame=drop_orphan_qc(frame),
                node_path=f"{self.node_path}/{match.station_id}",
                attrs=self.base_attrs(title=match.name, station_id=match.station_id),
                var_attrs={s.var: cf.cf_attrs(s, to_cf_units=to_cf) for s in specs.values()},
            ))
        return out
```

Register it:

```python
omnisea.register_provider(ShoreLoggerProvider())
omnisea.sources()   # -> [..., 'shorelogger_sst']
```

---

## Getting indexed automatically

Ship your provider as a package and declare an entry point. omnisea discovers it on import — no
user code, no registration call:

```toml
# pyproject.toml
[project.entry-points."omnisea.providers"]
shorelogger = "shorelogger.provider:ShoreLoggerProvider"
```

The entry point may name a `Provider` (registering all its sources) or a single `DataSource`.
A plugin that fails to import is logged and skipped — one broken package cannot take down the
registry.

If your source needs its own query knob, declare it so a typo raises instead of being ignored:

```python
omnisea.register_option("shorelogger_depth_m", "which logger depth to read")
```

---

## Checklist before you publish

- [ ] `discover()` is cheap — no bulk transfer, and returns `[]` rather than raising when
      there is nothing to offer.
- [ ] Anything `fetch()` needs from `discover()` is read with `match.require(...)`, so an
      adapter bug is loud rather than a silently missing station.
- [ ] Every `standard_name` appears in the CF standard name table; unmapped quantities use
      `standard_name=""` and a clear `long_name`.
- [ ] Units in the `units` attribute are the units the values are *actually* in — pass
      `self.to_cf_units(query)` to **both** `cf.convert` and `cf.cf_attrs` so they cannot
      disagree.
- [ ] Encoding repairs go in `scale`/`offset`; unit conversions go in `cf_scale`/`cf_offset`.
- [ ] Unmapped fields are carried, not dropped (`include_unmapped`).
- [ ] The returned frame is trimmed to the requested window — check what your upstream's time
      filter actually filters on. ECCC's `climate-hourly` filters on *local* dates while
      publishing UTC timestamps.
- [ ] Upstream interval caps are handled with `chunk_time(...)`, not by failing.
- [ ] QC flags are carried as `<var>_qc`, never silently dropped.
- [ ] Any variable that summarizes an interval carries `cell_methods`, so `align()` resamples
      it correctly instead of interpolating an accumulation.
- [ ] Predictions and models live under a different `node_path` than observations, so they can
      never be mistaken for measurements.
- [ ] Errors raised are `omnisea` errors (`UpstreamError`, `ProviderError`), so users can catch
      `omnisea.OmniseaError` and mean it. Let them propagate from `fetch()` — `Catalog.fetch()`
      decides whether to raise or collect, and swallowing them inside a source removes that
      choice from the caller.
- [ ] A rolling-window dataset declares `retention`, so a historical query gets an explanation
      instead of an empty result.

---

## Sources worth adding

The extension path from the design doc, in rough order of coverage won:

| Source | Base class | Notes |
|---|---|---|
| ERDDAP (`erddapy`) | `RetrievalSource` | Largest coverage win. `tabledap` reuses the point path; `griddap` returns a lazy `Dataset` into `/gridded/`. |
| NOAA CO-OPS | `RetrievalSource` | US tides; pairs with DFO across the border. |
| Copernicus Marine | `RetrievalSource` | First gridded source. Keep `subset()` lazy — no `.load()`. |
| STAC (`pystac-client`) | `DiscoverySource` | Discovery engine; contributes Catalog rows, not arrays. |
| OGC API - EDR | `RetrievalSource` | `Query` is already EDR-shaped, so this is close to a pass-through. |
| `eo_tides` | `RetrievalSource` | A *derived* node under `/predictions/tides_model/`, computed at query points. |
