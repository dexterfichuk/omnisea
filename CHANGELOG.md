# Changelog

## 0.1.0 — unreleased

First release.

### What it does

One Python client for marine data that is otherwise spread across a dozen incompatible APIs.
**18 data sources across 5 organizations** — DFO tide gauges, thirteen ECCC climate, weather,
buoy and hydrometric collections, any ERDDAP server worldwide, Ocean Networks Canada's cabled
observatories, and CIOOS metadata records.

- **One query shape** for every source: an area or a list of named sites, a time window, CF
  variable names. Deliberately EDR-shaped.
- **Discovery before download.** `discover()` returns a printable catalogue with row estimates;
  `fetch()` refuses anything over a ceiling, with the knob to change it in the message.
- **CF canonicalization** that never silently rescales. Values stay in provider units with the
  units recorded beside them; `to_cf_units=True` is opt-in. Encoding *repairs* — ECCC ships wind
  direction in tens of degrees — are always applied, because those are not a unit choice.
- **Nothing is dropped.** Fields with no CF mapping travel under the provider's own name, tagged
  `omnisea_mapped = 0`. QC flags are carried as `<var>_qc`.
- **`align()`** joins ragged sources onto one time axis, choosing each variable's resampling from
  its CF `cell_methods` rather than guessing: sums stay sums, extremes stay extremes, compass
  bearings are combined as unit vectors, and only instantaneous values are interpolated. Every
  choice is recorded in `attrs["omnisea_aggregation"]`.
- **`correlations()` / `drop_correlated()` / `model_matrix()`** turn a lossless tree into a
  defensible model matrix, and say what they removed and why.
- **`provenance()` / `citation()`** derive attribution from the result rather than from memory,
  including DOIs where a provider supplies them, and report what went wrong.
- **`xarray.DataTree`** output that round-trips losslessly to netCDF.

### Adding a source

A new data source is one new file implementing `discover()` and `fetch()`. Third-party providers
register through the `omnisea.providers` entry point and are indexed automatically. The contract
is executable: `python -m omnisea.conformance`.

### Known limitations

- Not on PyPI yet; install from the repository.
- Ocean Networks Canada requires a free API token (`onc_token=` or `ONC_TOKEN`). Every other
  source works without a credential.
- `climate-normals` is deliberately unsupported — it is not a time series.
- Antimeridian-crossing bounding boxes are refused rather than silently mishandled.
- Row estimates are cadence x window, not inventory-aware, so a part-time station can return
  fewer rows than its catalogue entry advertised.
- A single response body is capped at 512 MB (`omnisea.http.MAX_RESPONSE_BYTES`); a server
  streaming an endless body is refused rather than growing the process without bound.
- `radius_km` is capped at 2,000 km per site — a slipped decimal point otherwise becomes
  thousands of requests. Use `bbox=` for a genuinely global query.
- ERDDAP's `to_cf_units=True` is a no-op: those datasets state the units their numbers are in,
  not how to reach canonical CF units, and omnisea will not guess a scale factor for someone
  else's data.
