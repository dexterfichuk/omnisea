# Changelog

## 0.1.0 — first release

One Python client for marine data that is otherwise spread across a dozen incompatible APIs.
**18 data sources across 5 organizations** — DFO tide gauges, thirteen ECCC climate, weather,
buoy and hydrometric collections, any ERDDAP server worldwide, Ocean Networks Canada's cabled
observatories, and CIOOS metadata records.

### What it does

- **One query shape** for every source: an area or a list of named sites, a time window, CF
  variable names. Deliberately EDR-shaped.
- **Discovery before download.** `discover()` returns a printable catalogue with row estimates;
  `fetch()` refuses anything over a ceiling, with the knob to change it in the message.
- **CF canonicalization that never silently rescales.** Values stay in provider units with the
  units recorded beside them; `to_cf_units=True` is opt-in and leaves a trace on the variable.
  Encoding *repairs* — ECCC ships wind direction in tens of degrees — are always applied,
  because those are not a unit choice.
- **Nothing is dropped.** Fields with no CF mapping travel under the provider's own name,
  tagged `omnisea_mapped = 0`. QC flags are carried as `<var>_qc`.
- **`align()`** joins ragged sources onto one time axis, choosing each variable's resampling
  from its CF `cell_methods` rather than guessing: sums stay sums, extremes stay extremes,
  compass bearings are combined as unit vectors, and only instantaneous values are
  interpolated. Sources that label rows by local calendar date are read in the station's own
  time zone, on both the `on=` and `freq=` paths.
  Every choice is recorded in `attrs["omnisea_aggregation"]`, keyed by the column names the
  frame actually carries.
- **`correlations()` / `drop_correlated()` / `model_matrix()`** turn a lossless tree into a
  defensible model matrix, and say what they removed and why.
- **`provenance()` / `citation()`** derive attribution from the result — naming stations, DOIs
  where a provider supplies them, and what went wrong — rather than from memory.
- **US tides and rivers, natively.** `noaa_coops` mirrors `dfo_tides` branch for branch —
  six-minute observed water levels under `in_situ/tides`, predicted extrema under
  `predictions/tides_hilo`, with the vertical datum stated on the node and the variable
  (MLLW by default, IGLD on the Great Lakes, `coops_datum=` to choose). `usgs_water` serves
  US river discharge, stage and water temperature under the same branch as ECCC's gauges,
  excluding discontinued sites by their own period of record. A cross-border query produces
  one tree shape with both countries' licences attributed.
- **Citations print at fetch time, in full.** Every `fetch()` prints the complete citation
  block — organizations, stations, licences, terms — as the tree is handed over (`cite=False`
  silences it) and stamps the same block into `tree.attrs["citation"]`, which survives
  `to_netcdf()` and rides onto `align()`'s matrix as `omnisea_citation`. Attribution no
  longer depends on remembering a later call.
- **CDIP and UHSLC join as named ERDDAP servers.** Scripps' research wave buoys
  (spectral-grade, archives to the 1980s, dense on the US West Coast) and the University of
  Hawaii Sea Level Center's century-long research tide archive. Both are global-extent
  aggregate servers, so they are named-only (`sweep=False`). Reaching UHSLC fixed a general
  bug: datasets indexing longitude 0–360 answered west-negative constraints with "no
  matching results" — an empty answer indistinguishable from "no station here"; constraints
  now convert per the longitude variable's own `actual_range`, taking the larger half when
  a box straddles the seam.
- **`fetch(stations=...)` — address stations by id.** Seven of eight surveyed projects
  start from a station id they already know (a hardcoded gauge in a boat display, a river
  number in a model-forcing script) and none wants to supply a position.
  `fetch(stations={"noaa_coops": "9444090", "usgs_water": "12045500"}, time=...)` resolves
  each id to its own coordinates through the source's catalogue and returns exactly the
  named stations — four agencies, two countries, no coordinates in the call.
- **CO-OPS grew three capabilities.** Windows before 1996 read the `hourly_height` archive
  automatically (six-minute `water_level` begins in 1996; before this, older requests simply
  failed — the gap a GNSS-reflectometry project hit without learning why).
  `coops_high_low=True` fetches the *observed* tidal extrema into `in_situ/tides_extrema` —
  a measurement product despite its prediction-flavoured name. And every tide node now
  carries the full **datums ladder** (epoch, orthometric datum, metric offsets for
  MHHW…MLLW), turning "the datum is stated" into "the datum is convertible".
- **`usgs_parameters=` / `usgs_site_types=`** open NWIS beyond the curated codes (uncurated
  ones return under NWIS's own name, marked unmapped), `63680` turbidity joins the curated
  set, and `usgs_water_daily` serves mean, max and min with the `cell_methods` each earns.
- **`ndbc_stdmet` — waves.** No other source served significant wave height, period or
  direction natively; NDBC's ~1,900 buoys and shore stations do, plus marine wind, pressure
  and sea surface temperature, already SI, already UTC. Three file families (archived years,
  current-year months, 45-day realtime) stitch seamlessly, all-nines sentinels read as gaps,
  and a 404 year is an answer. The station table relays partner platforms — dozens of ECCC
  buoys — so the same source answers on both coasts of the border.
- **`usgs_water_daily`** — daily mean discharge, stage and water temperature, the partner to
  `eccc_hydrometric_daily`, labelled by local calendar date and read in station-local time the
  same way. NWIS discovery now counts only the record kinds each source can serve, so a site
  holding nothing but water-quality grab samples is no longer promised and fetched empty.
- **Constant text is stored once.** A mooring that repeats `scientist`, `project` and `agency`
  on every ten-minute row becomes a node whose metadata is scalar coordinates — one netCDF
  write went from 9.8 MB to under 1 MB with nothing lost.
- **Faster, and kinder to the servers.** Request concurrency is two-level — 4 per host, 24
  overall — so a bare discovery across ~26 sources at fifteen institutions is bounded by the
  slowest server rather than rationed through one pool of eight, while no single institution
  sees more than four connections. ERDDAP catalogue metadata participates in
  `enable_cache()`. Bare discovery: 8.2 s → 4.6 s warm.
- **Eleven ERDDAP installations by name.** `erddap_server="hakai"`, a list of names, or `"all"`
  — one adapter already read every ERDDAP, and this removes the part that needed you to know a
  URL. `omnisea.erddap_servers()` says what each one holds. A sweep survives one institution
  having a bad day and records what it could not reach; if none answer, it raises rather than
  reading as an empty ocean.
- **`xarray.DataTree`** output that round-trips losslessly to netCDF.
- **A tabledap response is split into one node per series**, using the dataset's own
  `cdm_timeseries_variables` and its vertical coordinate. A mooring reporting twelve depths on
  one clock becomes twelve nodes, not one column wandering through the water column; where
  nothing distinguishes two rows of an instant, omnisea says so rather than keeping one.

### Adding a source

One new file implementing `discover()` and `fetch()`. Third-party providers register through
the `omnisea.providers` entry point and are indexed automatically. The contract is executable:
`python -m omnisea.conformance`.

### Operational

- `set_timeout()`, `set_max_concurrency()`, `enable_cache()` and `clear_caches()` for
  unattended runs. A single response body is capped at 512 MB
  (`omnisea.http.MAX_RESPONSE_BYTES`).
- API tokens passed in a query string are redacted from log lines, from every error message,
  from response bodies that echo them back, and from the `source_url` written into saved files.
- `fetch()` raises when a source fails, in either the discovery or the retrieval phase; pass
  `on_error="collect"` to proceed with what answered.
- Fully typed, with `py.typed`.

### Testing

706 offline tests over committed real API responses, plus 55 live integration tests (3 need an
ONC token). CI runs the offline suite, ruff and the conformance checker on Python 3.11–3.13,
with a separate job pinned to the oldest supported dependency versions and a weekly live run to
catch upstream drift.

### Known limitations

- Not on PyPI yet; install from the repository (see the README's Installing section).
- Optional extras are `cache`, `netcdf`, `cioos` and `examples`. `erddap`, `cmems` and `stac`
  were declared during development and removed before release: erddapy is deliberately unused,
  and the other two were for providers that do not exist yet.
- Ocean Networks Canada requires a free API token (`onc_token=` or `ONC_TOKEN`). Every other
  source works without a credential, and ONC is skipped rather than failing when none is set.
- `climate-normals` is deliberately unsupported — it is not a time series.
- Antimeridian-crossing bounding boxes are refused rather than silently mishandled.
- `radius_km` is capped at 2,000 km per site; use `bbox=` for a genuinely global query.
- Row estimates are cadence x window, not inventory-aware, so a part-time station can return
  fewer rows than its catalogue entry advertised.
- ERDDAP's `to_cf_units=True` is a no-op: those datasets state the units their numbers are in,
  not how to reach canonical CF units, and omnisea will not guess a scale factor for someone
  else's data.
- A local-date source is read in the station's own time zone, daylight saving included, when
  the provider knows it — ECCC's daily and monthly collections resolve it from the station's
  province. Elsewhere the offset is estimated from longitude and rounded to the hour, which can
  be an hour out against a whole day of error if the stamp were read as UTC.
