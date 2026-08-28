# The omnisea Field Guide

One call against the ocean-observing systems of two countries. `omnisea` speaks to every
public ocean data source in the US and Canada — federal agencies, provincial networks,
research institutions, community ERDDAP servers — and returns one CF-compliant tree, whatever
it had to do to get there.

**44 sources · 19 organizations · 2 countries · 1 API**

```python
import omnisea

tree = omnisea.fetch(
    sites=omnisea.Site(48.298, -123.532, "Race Rocks", radius_km=60),
    time=("2024-07-01", "2024-07-08"),
)
```

## What it guarantees

These are the properties every source is held to — enforced by `python -m omnisea.conformance`
and the test suite, not by convention:

- **All times are UTC.** Sources publishing local time, local *dates* (ECCC's and USGS's
  daily archives), or DST-ambiguous stamps are converted or flagged with `time_reference` so
  `align()` can read them in station-local time.
- **Units ride with the numbers.** Values stay in the units the agency publishes — US feet
  beside Canadian metres — with `units` recorded on every variable. `to_cf_units=True`
  converts to SI at read time, so the attribute and the number can never disagree.
- **Observations and predictions never share a node.** A harmonic tide prediction and a
  measured water level look identical in a flat table; in the tree they live in
  `in_situ/` and `predictions/` branches respectively.
- **The vertical datum is stated** on tide nodes (`MLLW`, `MSL`, chart datum, `IGLD` on the
  Great Lakes) — on the node *and* the variable.
- **Empty answers are answers.** A station that matched but had no rows for your window is
  reported in `citation()`, not silently dropped; a source with nothing in your region
  returns nothing rather than an error.
- **Every resampling choice is auditable.** `align()` records the rule per column in
  `attrs["omnisea_aggregation"]` and the pre-resample observation count in
  `attrs["omnisea_samples"]` — the count interpolation cannot inflate.
- **Attribution is generated, not remembered.** `citation(tree)` names every organization,
  licence, and station that contributed.

## Sources by organization

| Organization | Sources | Notes |
|---|---|---|
| Fisheries and Oceans Canada (DFO/CHS) | `dfo_tides` | 1,573 tide gauges: observed, predicted, high/low events |
| Environment and Climate Change Canada | `eccc_climate`, `eccc_climate_daily`, `eccc_climate_monthly`, `eccc_climate_normals`, `eccc_hydrometric`, `eccc_hydrometric_daily`, `eccc_hydrometric_monthly`, `eccc_hydrometric_annual`, `eccc_hydrometric_annual_peaks`, `eccc_ahccd_annual`, `eccc_ahccd_monthly`, `eccc_ahccd_seasonal`, `eccc_swob` | Climate, rivers, homogenized archives, realtime SWOB |
| **NOAA CO-OPS** | `noaa_coops` | **US tide gauges, natively** — six-minute observations and predicted extrema, datum stated (MLLW/MSL/NAVD, IGLD on the Great Lakes), `coops_datum=` to choose |
| **NOAA NDBC** | `ndbc_stdmet` | **Buoys — where wave data lives**: significant height, period, direction, marine wind, pressure, SST; the station table relays partner buoys, ECCC's included |
| **USGS NWIS** | `usgs_water`, `usgs_water_daily` | **US river gauges** — discharge, stage, water temperature at native cadence and as daily means; period of record honoured |
| Ocean Networks Canada | `onc` | Cabled observatories; needs a free API token (`onc_token=`) |
| CIOOS (national + Pacific/Atlantic/SLGO) | `cioos` | Discovery across the national catalogue |
| Eleven ERDDAP installations | `cioos_pacific`, `cioos_atlantic`, `slgo`, `hakai`, `onc_erddap`, `ioos_sensors`, `glider_dac`, `osmc`, `salishseacast`, `nwem`, `polar_watch` (tabledap + griddap each) | Named first-class providers, region-gated, lazy grids |

Every source but ONC needs no credential at all.

## The workflow

```python
# 1. Look before you download — the catalogue is free
cat = omnisea.discover(sites=SITE, time=WINDOW)      # no providers= sweeps everything

# 2. Narrow: nearest station per source, or name what you want
tree = cat.filter(nearest=1).fetch()

# 3. One matrix, every resampling choice recorded
X = omnisea.align(tree, freq="1h")

# 4. Prune redundancy with an audit trail
X = omnisea.drop_correlated(X, threshold=0.9)        # judges by real observation counts

# 5. Attribution for the methods section
print(omnisea.citation(tree))
```

## Cross-border, one call

The Canada–US boundary runs down the middle of Juan de Fuca Strait, so both countries answer
one query:

```
                           node station_name  n_time
  /predictions/tides_hilo/07080   Pedder Bay      14     <- DFO, chart datum
         /in_situ/tides/9444090 Port Angeles    1681     <- NOAA, MLLW — stated on the node
/predictions/tides_hilo/9444090 Port Angeles      20
  /in_situ/hydrometric/12045500  ELWHA RIVER     673     <- USGS, ft³/s kept, m³/s on request
```

Each node carries its own country's licence; `citation(tree)` renders both.

## Where to go next

- [`examples/quickstart_model.ipynb`](../examples/quickstart_model.ipynb) — the 19-cell
  end-to-end demo: two countries → one model → pruned features → citation.
- [`examples/quickstart_everything.ipynb`](../examples/quickstart_everything.ipynb) — no
  `providers=` anywhere: every source in a region, every variable in one matrix, a model over
  all of it.
- The five regional notebooks (`bamfield`, `juan_de_fuca`, `strait_of_georgia`,
  `prince_rupert`, `calvert_island`) — deeper walkthroughs, executed against live APIs.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — adding a provider; `noaa.py` and `usgs.py` are
  the reference implementations.
