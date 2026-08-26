# omnisea

**A unified Python client for ocean, tidal, and weather data.**

Marine data access is fragmented across provider-specific APIs, each with its own field names,
units, time conventions and paging rules. omnisea is the middleware layer above them: pluggable
adapters per source, CF-standard canonicalization, EDR-shaped spatial-temporal queries, and
`xarray.DataTree` as the output container so 1-D point series and 4-D grids can coexist without a
lossy flat join.

```bash
pip install "omnisea[cache,netcdf] @ git+https://github.com/dexterfichuk/omnisea"
```

Not on PyPI yet, so install from the repository. To develop it:

```bash
git clone https://github.com/dexterfichuk/omnisea && cd omnisea
pip install -e ".[dev]"
```

Core dependencies are just `requests`, `pandas`, `numpy` and `xarray`; everything else —
response caching, netCDF output, the example notebook's plotting — is an optional extra, and
using a feature without its extra tells you exactly what to install. Python 3.11+; the test
suite runs in CI against both the latest and the oldest supported dependency versions.

**Want to see what it's like to use?** → **[examples/bamfield.ipynb](examples/bamfield.ipynb)**
is a complete walkthrough with output and plots already rendered, or run
[examples/bamfield.py](examples/bamfield.py) for the same thing in your terminal.

## The worked example

Bamfield Marine Sciences Centre sits on Barkley Sound, Vancouver Island. Ask what data exists
within 30 km of it:

```python
import omnisea

cat = omnisea.discover(lat=48.8353, lon=-125.1358, radius_km=30,
                       time=("2024-07-01", "2024-07-08"),
                       providers=["dfo_tides", "eccc_climate_daily", "eccc_hydrometric_daily"])
print(cat)
```

```
<Catalog: 12 station(s) from 3 source(s), ~886 rows>
                source station_id                             name  distance_km  n_rows_est                          variables
             dfo_tides      08545                         Bamfield         0.08         700 water_surface_height_above_refere…
eccc_hydrometric_daily    08HB076        SUGSAW LAKE NEAR BAMFIELD         3.31           8 water_surface_height_above_refere…
    eccc_climate_daily    1031316                 CAPE BEALE LIGHT         8.03           7 air_temperature, integral_wrt_tim…
    eccc_climate_daily    1035940                    PACHENA POINT        12.83           7 air_temperature, integral_wrt_tim…
eccc_hydrometric_daily    08HB048     CARNATION CREEK AT THE MOUTH        13.47           8 water_surface_height_above_refere…
             dfo_tides      08585                    Effingham Bay        13.81          28 water_surface_height_above_refere…
  ...

  (showing 6 of 12 columns; catalog.frame has them all: provider, site, lat, lon, first, last)
```

Drop the `providers=` argument to search all eighteen sources at once. The DFO tide gauge is
80 m from the research station — now pull the nearest station per source:

```python
tree = cat.filter(nearest=1).fetch()
print(omnisea.describe(tree))
```

```
                          node     station_name  n_time            start              end                                      variables
          /in_situ/tides/08545         Bamfield     672 2024-07-01 00:00 2024-07-08 00:00 water_surface_height_above_reference_datum, r…
 /predictions/tides_hilo/08545         Bamfield      27 2024-07-01 03:33 2024-07-07 22:03 water_surface_height_above_reference_datum_at…
/in_situ/weather_daily/1031316 CAPE BEALE LIGHT       8 2024-07-01 00:00 2024-07-08 00:00 air_temperature, air_temperature_min, air_tem…
```

`describe()` is the readable view; `summary()` returns the same rows with every column.

Two things arrived from that: what the gauge *measured*, and what the harmonic model
*predicts*. Plotted together, the predicted extrema land on the observed peaks:

![Observed tides at Bamfield with predicted extrema](docs/images/bamfield-tides.png)

That agreement is the real end-to-end check — it means the times, the units and the datum are
all right, not merely parseable. The network test suite asserts it.

Then save it, losslessly, as netCDF groups:

```python
omnisea.to_netcdf(tree, "bamfield.nc")   # or tree.to_netcdf(...) — same file
import xarray as xr
xr.open_datatree("bamfield.nc")          # the group structure round-trips
```

**Discovery is a separate step on purpose.** A marine query can quietly become enormous — ECCC's
`climate-hourly` collection reports 276 *million* rows unfiltered — so `discover()` returns a
printable estimate and `fetch()` refuses anything over a row ceiling, with the knob to change it
in the error message. Nothing large happens by accident.

## Many locations at once

A **`Site`** is the location type: coordinates, its own search radius, and the name you know it
by. That name travels with the results, so what comes back is joinable to your own records
without a second lookup.

`(lat, lon, name)` tuples, dicts, and DataFrames with `lat`/`lon` columns are accepted too — so
a CSV of moorings you already keep works without reshaping it first. Locations with no nearby
data simply contribute nothing, which is normal, not an error:

```python
from omnisea import Site

sites = [
    Site(48.8353, -125.1358, "Bamfield",   radius_km=25),
    Site(48.4204, -123.3656, "Victoria",   radius_km=25),
    Site(48.0000, -128.0000, "Open ocean", radius_km=25),   # nothing here
]

tree = omnisea.positions(sites, time=("2024-07-01", "2024-07-08"), nearest=1)

omnisea.coverage(tree)      # one row per requested site, INCLUDING the empty ones
omnisea.to_dataframe(tree)  # tidy: time, site, station_id, variable, value
```

`coverage()` reports every site you asked for, so gaps are visible rather than inferred from an
absence. Pass `group_by_site=True` to nest nodes as `/Bamfield/in_situ/tides/08545`.

![Tides at three sites from one query](docs/images/three-sites.png)

Bamfield and Tofino track each other closely — both open Pacific coast — while Victoria, inside
the Strait of Juan de Fuca, shows a markedly stronger diurnal inequality. One call, three
stations, three agencies' worth of plumbing you did not have to write.

## Bring your own data and build a model

The tree is lossless but ragged — tides every 15 minutes, climate summaries once a day, tidal
extrema at irregular turning points. A model wants a rectangle. `align()` produces one.

Say you have a field sheet: irregular sampling times and your own measurement. Join the
environmental data onto *your* timestamps:

```python
mine = pd.read_csv("field_samples.csv")     # time, chlorophyll_ug_L, secchi_m

tree = omnisea.fetch(sites=BAMFIELD, time=("2024-07-01", "2024-07-08"), nearest=1)
X = omnisea.align(tree, on=mine, tolerance="30min")
```

```
                     chlorophyll_ug_L  water_surface_height...  air_temperature  precipitation_amount
2024-07-01 09:14:00              5.64                    0.852             14.0                   0.0
2024-07-01 15:40:00              3.25                    2.212             14.0                   0.0
2024-07-02 10:05:00              1.77                    0.742             14.0                   0.0
2024-07-03 08:50:00              1.61                    1.864             13.8                   0.0
```

Your columns come through, so you have `y` and `X` in one table — straight into scikit-learn or
statsmodels. Or resample everything onto a regular grid instead:

```python
hourly = omnisea.align(tree, freq="1h")
weekly = omnisea.align(tree, freq="7D")
```

### Why this isn't just `.resample()`

Resampling a series wrongly is one of the easiest ways to publish a wrong result, and the right
answer differs per variable. omnisea doesn't guess — CF `cell_methods`, which every provider
already publishes, says what each variable *is*:

| `cell_methods` | Downsampling | Upsampling |
|---|---|---|
| `time: sum` (precipitation) | **sum** — a weekly total is the sum of daily totals | forward-fill; interpolating invents an intra-day distribution that was never measured |
| `time: maximum` (gust speed) | **max** — the max of maxima is a real maximum; their mean is a statistic of nothing | forward-fill |
| `time: mean` (daily mean temp) | mean | forward-fill — a daily mean spread across its own day is honest |
| none / `time: point` (tide height) | mean | **interpolate** — the only case where it's safe |

The same metadata governs the `on=` join. An **interval summary** matches *backwards within its
own interval* — a sample at 10:05 gets the total for the day containing it. An **instantaneous
reading** matches to the nearest observation within `tolerance`. Without that distinction, a
30-minute tolerance against a value stamped at midnight silently returns a column of `NaN`.

Every column records how it got there:

```python
X.attrs["omnisea_aggregation"]
```
```
water_surface_height_above_reference_datum@08545   nearest within 30min (8/8 matched)
air_temperature@1031316                            backward within its own 1d interval (8/8 matched)
precipitation_amount@1031316                       backward within its own 1d interval (8/8 matched)
water_surface_height..._at_extremum@08545          nearest within 30min (0/8 matched)
```

That last line is the point: what *couldn't* be matched is reported, not silently `NaN`.

### See what's redundant across sources, then drop it

A matrix assembled from many sources is collinear by construction — a station's mean, minimum
and maximum temperature share one week of weather, `heating_degree_days` is *derived from* mean
temperature, and an observed water level and its harmonic prediction are nearly one column. A
model fed all of them still predicts, but OLS splits the true effect arbitrarily among the
near-copies and the coefficients stop meaning anything.

```python
omnisea.correlations(X)                        # pairs that move together, strongest first
X = omnisea.drop_correlated(X, threshold=0.95)
X.attrs["omnisea_dropped"]                     # what went, and why
```

`correlations()` is the evidence: each pair's `r` comes with `n`, the overlapping samples it
was computed on, because an r of 1.0 over three points is noise wearing a convincing costume
(compass directions are excluded — a linear r between bearings is meaningless, the same reason
`align()` combines them as unit vectors). `drop_correlated()` removes only near-duplicates,
keeps the better-covered column of each pair, and records every removal — auditable like the
resampling choices. Columns you carried through `align(on=...)` are **never dropped and never
cause a drop** (correlation with *your* response is signal, not redundancy), and `keep=` pins
the copy you trust so its partners go instead.

### Then build a model

`examples/bamfield.ipynb` runs this end to end: a mock five-day logger deployment sampling every
two hours, joined to the real tide and climate series, then a regression.

The samples are **synthetic** — generated from the real fetched drivers with coefficients we
choose, plus noise — which makes it a test of the *pipeline*. If the join were misaligned by the
station's UTC offset, or a daily total had been interpolated, the fit would not recover them:

```
                                            fitted  true  error
intercept                                    5.969  6.00 -0.031
water_surface_height_above_reference_datum  -0.450 -0.45  0.000
air_temperature_max                          0.301  0.30  0.001
hour_sin                                     0.586  0.60 -0.014
```

![Observed vs predicted](docs/images/model-fit.png)

### Putting your data in the tree

If your measurements are the thing being modelled and you want one object holding response and
predictors together — round-trippable to netCDF, with your provenance beside the providers':

```python
tree = omnisea.add_local(tree, mine, name="Chlorophyll grab samples",
                         lat=48.8353, lon=-125.1358, station_id="BAM-CHL",
                         var_attrs={"chlorophyll_ug_L": {"units": "ug L-1"}})
```

A `cell_methods` you supply in `var_attrs` is honoured by `align()` exactly as a provider's
would be.

## What you get back

A plain `xarray.DataTree` — not a subclass, so it interoperates with the whole ecosystem.
Conveniences are module functions: `omnisea.summary(tree)`, `omnisea.to_dataframe(tree)`,
`omnisea.stations(tree)`, `omnisea.coverage(tree)`, `omnisea.fields(tree)`.

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

**Nothing is silently dropped — every field always comes back.** Fields with a CF equivalent are
renamed and described; fields without one travel under the provider's own name, tagged
`omnisea_mapped = 0`. SWOB ships about 74 fields and omnisea names 12 of them; the other 62 —
battery voltage, solar panel current, 24-hour rainfall totals — arrive too. QC flags are carried
as `<var>_qc`, never discarded.

`variables=` chooses **which sources and stations to fetch**, and is deliberately *not* a
projection over what they return. The GeoJSON response already carries every property, so
dropping columns would discard data that has already crossed the network:

```python
tree = omnisea.fetch(..., variables=["air_temperature"])
omnisea.fields(tree)     # -> 58 variables, from the 2 sources that have air_temperature
```

A name omnisea doesn't recognise doesn't get you an error, either — it keeps every source in
play, because a curated table is a floor, not an inventory:

```python
omnisea.fetch(..., variables=["batry_volt"])   # not a CF name; SWOB still gets queried
```

```python
omnisea.variables()      # the CF names omnisea curates — a floor, not an inventory
omnisea.fields(tree)     # what a particular fetch actually returned
```

## Sources

| Source | Provider | Data |
|---|---|---|
| `dfo_tides` | `dfo` | Water levels: observed, predicted, and high/low events (1573 stations) |
| `eccc_climate` | `eccc` | Hourly surface climate observations |
| `eccc_climate_daily` | `eccc` | Daily climate summaries |
| `eccc_hydrometric` | `eccc` | Realtime water level and river discharge (~30 days) |
| `eccc_hydrometric_daily` | `eccc` | Daily mean level and discharge — the historical archive |
| `eccc_hydrometric_monthly` | `eccc` | Monthly mean level and discharge |
| `eccc_hydrometric_annual` | `eccc` | Annual extremes, with the date each occurred |
| `eccc_hydrometric_annual_peaks` | `eccc` | Instantaneous annual peaks |
| `eccc_climate_monthly` | `eccc` | Monthly climate summaries |
| `eccc_ahccd_monthly` / `_seasonal` / `_annual` | `eccc` | Adjusted & Homogenized Canadian Climate Data — the long homogenized record |
| `eccc_swob` | `eccc` | Surface weather observations, realtime (~30 days) |
| `eccc_swob_marine` | `eccc` | Moored buoys: waves, sea surface temperature (~30 days) |
| `erddap_tabledap` | `erddap` | Any ERDDAP server's station/point datasets |
| `erddap_griddap` | `erddap` | Any ERDDAP server's gridded datasets, returned lazily |
| `onc_scalardata` | `onc` | Ocean Networks Canada cabled observatories and moorings — **needs a token** |
| `cioos_metadata` | `cioos` | CIOOS metadata records — discovery only, contributes catalogue rows |

Select a single dataset or a whole organization:

```python
omnisea.fetch(..., providers="dfo_tides")   # one dataset
omnisea.fetch(..., providers="eccc")        # every ECCC dataset, all thirteen
```

### Any ERDDAP server

ERDDAP is the same software running at IOOS, CIOOS, NOAA CoastWatch, Hakai, EMODnet and a few
hundred other institutions, so **one adapter reaches all of them** — point it at a different
server and that institution's whole catalogue is queryable with no new code:

```python
omnisea.fetch(lat=48.8353, lon=-125.1358, radius_km=30, time=("2024-07-01", "2024-07-08"),
              providers="erddap_tabledap",
              erddap_server="https://catalogue.hakai.org/erddap")
```

| Server | `erddap_server=` |
|---|---|
| IOOS Sensors (default) | `https://erddap.sensors.ioos.us/erddap` |
| DFO Pacific / CIOOS | `https://data.cioospacific.ca/erddap` |
| Hakai Institute | `https://catalogue.hakai.org/erddap` |
| NOAA CoastWatch | `https://coastwatch.pfeg.noaa.gov/erddap` |
| SalishSeaCast | `https://salishsea.eos.ubc.ca/erddap` |
| NW Environmental Moorings | `https://nwem.apl.uw.edu/erddap` |

**No field table is hardcoded.** Each dataset publishes its own `standard_name`, `units`,
`cell_methods` and QARTOD `ancillary_variables`, and omnisea reads that rather than replacing
it — so `align()` resamples an ERDDAP variable by the author's own metadata. `tabledap` joins
the point path (a table holding many platforms is split per station rather than collapsed);
`griddap` returns a **lazy** dataset, so a decade of a global analysis can sit in your tree and
you pay only for the pixels you read.

Discovery consults both `allDatasets` and the search index and unions them, because they
disagree in practice. A dataset whose publisher declared no extent is invisible to both — name
it directly to bypass the metadata filters:

```python
omnisea.fetch(..., erddap_datasets=["PRIMED_wavebuoy"], erddap_server=...)
```

If a named dataset can't be served as a time series — CIOOS Pacific's `IOS_P26_Annualized` is
indexed by an integer `Year` column and publishes no `time` variable at all — you get an
`omnisea` error saying so, not an upstream `400`.

### Ocean Networks Canada

ONC runs cabled observatories, moorings and autonomous platforms off both Canadian coasts and in
the Arctic — 1,992 locations, 219 measured properties. It is the one built-in source that needs
a credential. Register at [data.oceannetworks.ca](https://data.oceannetworks.ca), then
**Profile → Web Services API**:

```python
omnisea.fetch(lat=48.8145, lon=-125.2825, radius_km=6, time=("2024-07-01", "2024-07-02"),
              providers="onc_scalardata", onc_token="…")     # or export ONC_TOKEN=…
```

```
                  node    station_name  n_time  variables
 /in_situ/onc/FGPD/CTD     Folger Deep    1440  sea_water_temperature, sea_water_pressure, …
/in_situ/onc/FGPPN/CTD Folger Pinnacle    1440  sea_water_temperature, sea_water_practical_salinity, …
```

**Your token stays out of everything omnisea writes down.** ONC authenticates with a `?token=`
query parameter, and a URL like that would otherwise reach the debug log, the message of every
error, and the `source_url` recorded on each node — which gets written into netCDF files people
share and commit. All three are redacted, and so is the token ONC helpfully echoes back *inside*
its own error bodies.

One node per (location, instrument), because a location can carry a CTD and a hydrophone and a
current meter and those are not one time series. Raw ONC data can be 1 Hz — a day of one CTD is
86,400 rows per sensor — so requests are resampled server-side to a minute by default; pass
`onc_resample_seconds=0` for raw, or any interval you like.

ONC returns the exact citation it wants credited plus a **resolvable DOI** per deployment, and
omnisea carries both onto the node rather than inventing its own attribution:

```python
omnisea.citation(tree)
# Ocean Networks Canada Society. 2023. Folger Deep Conductivity Temperature Depth
# Deployed 2023-09-15. https://doi.org/10.34943/5629c51a-f715-44fe-b418-a289048c56fd
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

## Where did this data come from?

Every node carries the institution that published it, the licence and the URL it was read from,
so attribution comes out of the result rather than out of memory:

```python
omnisea.provenance(tree)     # one row per source: institution, licence, terms, stations, span
omnisea.citation(tree)       # an attribution block for a methods section
```

```
Data were retrieved with omnisea 0.1.0 on 2026-08-25 from 2 source(s) across 2 station(s),
covering 2024-07-01 to 2024-07-08.

  Fisheries and Oceans Canada / Canadian Hydrographic Service — dfo_tides (1 station(s)).
    Licence: Open Government Licence – Canada. Terms: https://open.canada.ca/...
  Environment and Climate Change Canada — eccc_climate_daily (1 station(s)).
    Licence: Open Government Licence – Canada. Terms: https://eccc-msc.github.io/...
```

It also reports what went *wrong* — an incomplete fetch, or stations that matched but returned
no rows. Publishing a partial pull without noticing is the failure this is meant to prevent.
Pass `include_urls=True` for the exact endpoint each series came from, which matters for
realtime sources whose contents cannot be recovered later from the query alone.

## Adding your own source

A new data source is **one new file**, not a refactor. omnisea models two levels — a `Provider`
is an organization (licence, base URL, auth); a `DataSource` is one queryable dataset (fields,
node path, discover/fetch). Implement the interface and your source is discovered, filtered,
CF-described and assembled alongside every built-in one.

**→ [docs/adding-a-provider.md](docs/adding-a-provider.md)** is the full contract.
**→ [examples/provider_template.py](examples/provider_template.py)** is a runnable source with
every hook commented — copy it rather than starting from scratch.
**→ [CONTRIBUTING.md](CONTRIBUTING.md)** is the PR path.

The contract is also **a program**, so a pull request self-checks:

```bash
python -m omnisea.conformance     # every registered source
```

```python
omnisea.check_source(MySource(MyProvider()))
```

It checks the things where being wrong is *quiet*: a `standard_name` that is not in the CF
table (bundled, so it works offline), two fields colliding on one output variable, a value
converted with no `cf_units` to convert to, an aggregate with no `cell_methods` for `align()`
to read. Written on a Tuesday, it found two real defects and one library-wide bug the same day.

Ship it as a package and it is indexed automatically:

```toml
[project.entry-points."omnisea.providers"]
my_org = "my_package.provider:MyOrgProvider"
```

## Types

Two small types carry meaning that a bare tuple would lose:

```python
from omnisea import Site

site = Site(48.8353, -125.1358, "Bamfield Marine Sciences Centre", radius_km=30)
site.label                      # 'Bamfield Marine Sciences Centre' — travels onto the data
omnisea.fetch(sites=site, time=...)
```

`BBox` is a `NamedTuple` in the OGC order `(west, south, east, north)`, so `bbox.south` beats
`bbox[1]` and a reader can tell which convention is in play — lon-first, not the lat-first order
people often reach for. It still unpacks, indexes and compares as an ordinary tuple, so plain
tuples remain acceptable input everywhere:

```python
cat = omnisea.discover(bbox=(-125.22, 48.78, -125.05, 48.90), time=...)
cat.query.bbox.south      # 48.78
cat.query.bbox.centre     # (48.84, -125.135) as (lat, lon)
```

A lat-first bbox is rejected rather than silently swapped.

## Caching

Off by default. Turn it on and repeat queries stop re-downloading the parts that don't change:

```bash
pip install "omnisea[cache]"
```
```python
omnisea.enable_cache(path="~/.omnisea-cache.sqlite")
```

What gets cached is decided per endpoint rather than by one TTL, because they differ in kind.
Station catalogues and metadata are near-static (7 days); GitHub-hosted CIOOS records get a day
(unauthenticated GitHub allows 60 requests an hour, which is the difference between discovery
working twice in a row and not); appended archives get an hour. **Measurement endpoints are
never cached**, and are excluded explicitly rather than by omission, so the exclusion survives a
caller who passes `expire_after=` to cache everything else.

That judgment is part of the provider contract, not a core table: each provider declares a
`cache_policy` for its own endpoints, and third-party providers' rules merge in exactly like the
built-ins'. An endpoint nobody has claimed is never cached.

One thing worth knowing: IWLS answers *every* request — including its near-static 1573-station
list — with `Cache-Control: no-cache, no-store, must-revalidate`. Honouring that would reduce
the feature to a no-op, so omnisea overrides it deliberately for the endpoints its policy
considers static.

## When a source can't help

Several ECCC collections are rolling archives — `swob-realtime` and `hydrometric-realtime` keep
roughly 30 days. Asking them for last year returns nothing, and "nothing" reads as *there is no
station here*, which is a different and wrong conclusion. So sources declare their retention and
omnisea says what happened:

```
<Catalog: no stations found for 1 site(s) in Query(position(48.8353, -125.1358, r=30.0km), ...)>
  - eccc_hydrometric: holds only the last ~30 days (back to 2026-07-26); the requested
    window ends 2024-07-08 and is entirely outside it
```

Discovery and retrieval treat failure differently, on purpose. `discover()` collects per-source
errors and carries on — it is a survey, and the Catalog shows you which sources are missing from
it. `fetch()` **raises** by default, because it produces the data you will analyse and a tree
quietly missing a source looks exactly like a tree where that source had nothing to say. Pass
`on_error="collect"` for exploratory work; the failures are then recorded in the tree's
`omnisea_fetch_errors` attribute rather than dropped.

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
pytest -m "not network"   # 670 offline tests over committed real API responses
pytest -m network         # 55 live integration tests (3 need ONC_TOKEN)
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
