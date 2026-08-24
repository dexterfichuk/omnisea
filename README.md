# omnisea

**A unified Python client for ocean, tidal, and weather data.**

Marine data access is fragmented across provider-specific APIs, each with its own field names,
units, time conventions and paging rules. omnisea is the middleware layer above them: pluggable
adapters per source, CF-standard canonicalization, EDR-shaped spatial-temporal queries, and
`xarray.DataTree` as the output container so 1-D point series and 4-D grids can coexist without a
lossy flat join.

```bash
pip install -e ".[dev]"
```

**Want to see what it's like to use?** → **[examples/bamfield.ipynb](examples/bamfield.ipynb)**
is a complete walkthrough with output and plots already rendered, or run
[examples/bamfield.py](examples/bamfield.py) for the same thing in your terminal.

## The worked example

Bamfield Marine Sciences Centre sits on Barkley Sound, Vancouver Island. Ask what data exists
within 30 km of it:

```python
import omnisea

cat = omnisea.discover(lat=48.8353, lon=-125.1358, radius_km=30,
                       time=("2024-07-01", "2024-07-08"))
print(cat)
```

```
            source station_id                         name  distance_km  n_rows_est
         dfo_tides      08545                     Bamfield     0.079201         700
eccc_climate_daily    1031316             CAPE BEALE LIGHT     8.031310           7
eccc_climate_daily    1035940                PACHENA POINT    12.827256           7
  eccc_hydrometric    08HB048 CARNATION CREEK AT THE MOUTH    13.473495         672
  eccc_hydrometric    08HB014   SARITA RIVER NEAR BAMFIELD    13.731407         672
         dfo_tides      08585                Effingham Bay    13.806121          28
```

The DFO tide gauge is 80 m from the research station. Now pull the nearest station per source:

```python
tree = cat.filter(nearest=1).fetch()
print(omnisea.summary(tree))
```

```
                          node     station_name  n_time  variables
          /in_situ/tides/08545         Bamfield     672  water_surface_height_above_reference_datum, reviewed
 /predictions/tides_hilo/08545         Bamfield      27  water_surface_height_above_reference_datum_at_extremum, reviewed
/in_situ/weather_daily/1031316 CAPE BEALE LIGHT       8  air_temperature, air_temperature_min, air_temperature_max, ...
```

Two things arrived from that: what the gauge *measured*, and what the harmonic model
*predicts*. Plotted together, the predicted extrema land on the observed peaks:

![Observed tides at Bamfield with predicted extrema](docs/images/bamfield-tides.png)

That agreement is the real end-to-end check — it means the times, the units and the datum are
all right, not merely parseable. The network test suite asserts it.

Then save it, losslessly, as netCDF groups:

```python
tree.to_netcdf("bamfield.nc")
import xarray as xr
xr.open_datatree("bamfield.nc")     # the group structure round-trips
```

**Discovery is a separate step on purpose.** A marine query can quietly become enormous — ECCC's
`climate-hourly` collection reports 276 *million* rows unfiltered — so `discover()` returns a
printable estimate and `fetch()` refuses anything over a row ceiling, with the knob to change it
in the error message. Nothing large happens by accident.

## Many locations at once

Hand it a list of sites — `Site` objects, `(lat, lon, name)` tuples, dicts, or a DataFrame with
`lat`/`lon` columns, so a CSV of moorings works directly. Locations with no nearby data simply
contribute nothing, which is normal, not an error:

```python
sites = [
    {"name": "Bamfield",  "lat": 48.8353, "lon": -125.1358},
    {"name": "Victoria",  "lat": 48.4204, "lon": -123.3656},
    {"name": "Open ocean","lat": 48.0000, "lon": -128.0000},   # nothing here
]

tree = omnisea.positions(sites, radius_km=25, time=("2024-07-01", "2024-07-08"), nearest=1)

omnisea.coverage(tree)      # one row per requested site, INCLUDING the empty ones
omnisea.to_dataframe(tree)  # tidy: time, site, station_id, variable, value
```

`coverage()` reports every site you asked for, so gaps are visible rather than inferred from an
absence. Pass `group_by_site=True` to nest nodes as `/Bamfield/in_situ/tides/08545`.

![Tides at three sites from one query](docs/images/three-sites.png)

Bamfield and Tofino track each other closely — both open Pacific coast — while Victoria, inside
the Strait of Juan de Fuca, shows a markedly stronger diurnal inequality. One call, three
stations, three agencies' worth of plumbing you did not have to write.

## What you get back

A plain `xarray.DataTree` — not a subclass, so it interoperates with the whole ecosystem.
Conveniences are module functions: `omnisea.summary(tree)`, `omnisea.to_dataframe(tree)`,
`omnisea.stations(tree)`, `omnisea.coverage(tree)`.

Each node is a CF discrete-sampling-geometry `timeSeries`: a `time` dimension, scalar
`latitude`/`longitude`/`station_id`/`station_name` coordinates, and attributes recording
`featureType`, `Conventions`, provider, licence, source URL and datum. The root records the query.

Observations and predictions live in **separate branches** (`/in_situ/tides/` vs
`/predictions/tides_hilo/`), because a harmonic tide prediction and a measurement look identical
in a flat table and are not remotely the same thing.

## Two promises about your numbers

**Nothing is silently rescaled.** Values stay in the units the provider published, and the
`units` attribute says which those are. Ask for canonical CF units explicitly:

```python
tree = omnisea.fetch(..., to_cf_units=True)   # degC -> K, km/h -> m/s, kPa -> Pa
```

Encoding *repairs* are different and always applied — ECCC ships wind direction in tens of
degrees, so a raw `25` means 250°. That is a storage encoding, not a unit choice.

**Nothing is silently dropped.** Fields with a CF equivalent are renamed and described; fields
without one travel under the provider's own name, tagged `omnisea_mapped = 0`. A SWOB station
returns `air_temperature` *and* `batry_volt`. QC flags are carried as `<var>_qc`, never discarded.

```python
omnisea.variables()   # the CF names available, and which source serves each
```

## Sources

| Source | Provider | Data |
|---|---|---|
| `dfo_tides` | `dfo` | Water levels: observed, predicted, and high/low events (1573 stations) |
| `eccc_climate` | `eccc` | Hourly surface climate observations |
| `eccc_climate_daily` | `eccc` | Daily climate summaries |
| `eccc_swob` | `eccc` | Surface weather observations, realtime (~30 days) |
| `eccc_hydrometric` | `eccc` | Realtime water level and river discharge |
| `cioos_metadata` | `cioos` | CIOOS metadata records — discovery only, contributes catalogue rows |

Select a single dataset or a whole organization:

```python
omnisea.fetch(..., providers="dfo_tides")   # one dataset
omnisea.fetch(..., providers="eccc")        # all four ECCC datasets
```

### CIOOS metadata records

`cioos_metadata` reads records authored in the
[CIOOS metadata-entry-form](https://github.com/cioos-siooc/metadata-entry-form). It is a
*discovery* source: a record says a dataset exists, where and when, which Essential Ocean
Variables it covers, and where to download it — it hands back a URL, not an array.

There is no single public endpoint (the form publishes each organization's records to that
organization's own GitHub repo, and its Firebase database is not world-readable), so point it at
wherever you keep them:

```python
cat = omnisea.discover(bbox=(-125.3, 48.7, -125.0, 49.0), time=("2024-01-01", "2024-12-31"),
                       cioos_records="cioos-siooc/records")   # or a path, directory, or URL
cat.metadata()    # titles, extents, EOVs, licences, download URLs
```

GOOS EOV tags are translated to CF standard names, so `variables=["sea_surface_temperature"]`
matches a record tagged `seaSurfaceTemperature`.

## Adding your own source

A new data source is **one new file**, not a refactor. omnisea models two levels — a `Provider`
is an organization (licence, base URL, auth); a `DataSource` is one queryable dataset (fields,
node path, discover/fetch). Implement the interface and your source is discovered, filtered,
CF-described and assembled alongside every built-in one.

**→ [docs/adding-a-provider.md](docs/adding-a-provider.md)** is the full contract, with a
complete worked example in [examples/csv_stations.py](examples/csv_stations.py) that the test
suite exercises.

Ship it as a package and it is indexed automatically:

```toml
[project.entry-points."omnisea.providers"]
my_org = "my_package.provider:MyOrgProvider"
```

## Design notes

`Query` is deliberately EDR-shaped — `bbox` or sites with radii, a UTC-normalized interval, CF
variable names, optional depth — so a native OGC API - EDR adapter is close to a pass-through.
`DataSource.fetch()` may return either a `StationSeries` (point path) or a ready `xarray.Dataset`
(gridded path); that union is the seam that lets Copernicus or ERDDAP `griddap` drop in without
touching the point-series assembly.

A few upstream traps are handled, each verified against the live services:

- **IWLS window caps are resolution-dependent** — 7 days at `ONE_MINUTE`, 31 days at every
  coarser resolution. Requests chunk to whichever applies, with boundary rows de-duplicated.
- **`climate-stations` publishes `LATITUDE` as integer micro-degrees** (`483300000`), so
  coordinates always come from `geometry.coordinates`.
- **`climate-hourly` filters on `LOCAL_DATE` while publishing `UTC_DATE`**, so a UTC window
  comes back shifted by the station's offset. omnisea pads the request and trims to the window
  you actually asked for.
- **`climate-daily` has no `UTC_DATE` at all.** Daily aggregates keep their local calendar date,
  stamped at `00:00Z`, with the convention recorded in the node's `time_reference` attribute.
  Inventing a UTC instant for a daily statistic would imply precision the data lacks.
- **SWOB units are read from each field's `-uom` sibling**, never hardcoded.

`sea_surface_height_above_reference_datum` — the name that reads as the obvious one for tide
gauge data — **is not in the CF standard name table**. omnisea emits the real name,
`water_surface_height_above_reference_datum`, and accepts the other as an input alias. A network
test validates every emitted `standard_name` against the published table.

## Examples

| File | What it shows |
|---|---|
| [examples/bamfield.ipynb](examples/bamfield.ipynb) | The full walkthrough, executed, with plots |
| [examples/bamfield.py](examples/bamfield.py) | The same walkthrough as a terminal script |
| [examples/csv_stations.py](examples/csv_stations.py) | A complete third-party provider in ~100 lines |

## Tests

```bash
pytest -m "not network"   # 161 unit tests over committed real API responses
pytest -m network         # 21 live integration tests
```

The network suite covers the edge cases fixtures cannot: the IWLS interval caps and chunk
stitching, pygeoapi paging past one page, stations correctly excluded for no time overlap, the
netCDF round-trip, and CF vocabulary validation.

It also checks the *physics*, not just the parsing: the observed series must be semidiurnal
(~2 highs/day), water levels must be physically plausible, and the independently-fetched
predicted extrema must line up with the observed peaks — currently within a median of 18 minutes
and 0.044 m.

## Licence

MIT. Data licences belong to the providers and are recorded on every node.
