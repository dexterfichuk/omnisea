"""A full walkthrough: finding and pulling data around Bamfield Marine Sciences Centre.

Bamfield sits on Barkley Sound, on the exposed west coast of Vancouver Island. Within 30 km
there is a DFO tide gauge (80 m from the station), two lighthouse climate records, and two
hydrometric gauges on nearby rivers — four different agencies' formats, one query.

Run it::

    python examples/bamfield.py

Everything here hits live APIs and takes about 20 seconds.
"""

from __future__ import annotations

import warnings

import pandas as pd

import omnisea

warnings.simplefilter("ignore")

BAMFIELD = dict(lat=48.8353, lon=-125.1358, radius_km=30)
WEEK = ("2024-07-01", "2024-07-08")


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
rule("1. What can omnisea talk to?")

print("organizations:", omnisea.providers())
print("datasets:     ", omnisea.sources())

variables = omnisea.variables()
print(f"\n{variables['standard_name'].nunique()} CF standard names across "
      f"{variables['source'].nunique()} sources. A sample:")
print(variables[["variable", "units", "source"]].drop_duplicates("variable").head(8)
      .to_string(index=False))
print("\n(Fields with no CF name are still returned — under the provider's own name.)")


# ---------------------------------------------------------------------------
rule("2. Discover: what exists near Bamfield, and how big is it?")

catalog = omnisea.discover(**BAMFIELD, time=WEEK)

print(f"{len(catalog)} stations from {len(catalog.sources)} sources, "
      f"~{catalog.n_rows_est:,} rows estimated\n")
print(catalog.frame[["source", "station_id", "name", "distance_km", "n_rows_est"]]
      .to_string(index=False))

print("\nNothing has been downloaded yet — this is the look-before-you-leap step.")
print("An unfiltered ECCC hourly query would match 276 MILLION rows; the estimate is")
print("how you find that out before making the request rather than after.")


# ---------------------------------------------------------------------------
rule("3. Narrow it down")

print("Nearest station per source:")
nearest = catalog.filter(nearest=1)
print(nearest.frame[["source", "station_id", "name", "distance_km"]].to_string(index=False))

print("\nOr slice by whatever matters to you:")
print(f"  within 10 km ........ {len(catalog.filter(max_distance_km=10))} stations")
print(f"  tide gauges only .... {len(catalog.filter(source='dfo_tides'))} stations")
print(f"  all ECCC datasets ... {len(catalog.filter(provider='eccc'))} stations")
print(f"  air temperature ..... {len(catalog.filter(variables='air_temperature'))} stations")


# ---------------------------------------------------------------------------
rule("4. Fetch")

tree = nearest.fetch()
print(omnisea.summary(tree)[["node", "station_name", "n_time", "start", "end"]]
      .to_string(index=False))

print("\nObservations and predictions are in SEPARATE branches:")
for node in tree.subtree:
    if node.dataset.data_vars:
        print(f"  {node.path}")
print("\nA harmonic prediction and a measurement look identical in a flat table.")
print("They are not the same thing, so they never share a node.")


# ---------------------------------------------------------------------------
rule("5. Look inside a node")

tides = tree["/in_situ/tides/08545"].dataset
print(tides)

print("\nProvenance travels with the data:")
for key in ("institution", "license", "datum", "datum_offset_CGVD2013", "source_url"):
    print(f"  {key:24s} {tides.attrs.get(key)}")


# ---------------------------------------------------------------------------
rule("6. Nothing is silently dropped")

mapped, carried, flags = [], [], []
for node in tree.subtree:
    for name, var in node.dataset.data_vars.items():
        if str(name).endswith("_qc"):
            flags.append(str(name))
        elif var.attrs.get("omnisea_mapped") == 0:
            carried.append(str(name))
        else:
            mapped.append(str(name))

print(f"CF-named ({len(mapped)}):")
print("   ", ", ".join(sorted(set(mapped))))
print(f"\ncarried under the provider's own name ({len(set(carried))}):")
print("   ", ", ".join(sorted(set(carried))))
print(f"\nQC flags kept alongside their variable ({len(set(flags))}):")
print("   ", ", ".join(sorted(set(flags))))
print("\nA realtime SWOB station adds dozens more — battery voltage, solar panel current,")
print("24-hour rainfall totals. omnisea canonicalizes what it can and keeps what it cannot.")

weather = tree["/in_situ/weather_daily/1031316"].dataset
print("\nAnd nothing is silently rescaled — units say what the values actually are:")
print(f"  air_temperature units = {weather['air_temperature'].attrs.get('units')!r}"
      f"  (pass to_cf_units=True for {weather['air_temperature'].attrs.get('cf_units')!r})")


# ---------------------------------------------------------------------------
rule("7. Tidy frame for analysis")

frame = omnisea.to_dataframe(tree)
print(frame.head(6)[["time", "station_id", "variable", "value"]].to_string(index=False))
print(f"\n{len(frame):,} rows, {frame['variable'].nunique()} variables, "
      f"{frame['station_id'].nunique()} stations — ready to plot or join.")


# ---------------------------------------------------------------------------
rule("8. Does it actually behave like the ocean?")

wlo = tides["water_surface_height_above_reference_datum"].to_series()
hilo = tree["/predictions/tides_hilo/08545"].dataset[
    "water_surface_height_above_reference_datum_at_extremum"].to_series()

values = wlo.values
is_peak = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
days = (wlo.index[-1] - wlo.index[0]) / pd.Timedelta(days=1)

print(f"observed range      : {values.min():.2f} to {values.max():.2f} m above chart datum")
print(f"highs per day       : {is_peak.sum() / days:.2f}  (semidiurnal coast expects ~2)")

peak_times = wlo.index[1:-1][is_peak]
predicted_highs = hilo[hilo > hilo.median()]
offsets = [min(abs((p - h).total_seconds()) / 60 for p in peak_times)
           for h in predicted_highs.index]
print(f"predicted highs     : {len(predicted_highs)}, median "
      f"{pd.Series(offsets).median():.0f} min from an observed peak")

interpolated = wlo.reindex(wlo.index.union(hilo.index)).interpolate().reindex(hilo.index)
print(f"height agreement    : mean |predicted - observed| = "
      f"{(interpolated - hilo).abs().mean():.3f} m, r = {interpolated.corr(hilo):.4f}")
print("\nTwo independently fetched series agreeing this closely is the real end-to-end check:")
print("it means the times, the units and the datum are all right, not merely parseable.")


# ---------------------------------------------------------------------------
rule("9. Many locations at once")

sites = [
    {"name": "Bamfield", "lat": 48.8353, "lon": -125.1358},
    {"name": "Tofino", "lat": 49.1530, "lon": -125.9066},
    {"name": "Victoria", "lat": 48.4204, "lon": -123.3656},
    {"name": "Open ocean", "lat": 47.5000, "lon": -128.5000},  # deliberately empty
]

multi = omnisea.positions(sites, radius_km=15, time=WEEK, providers="dfo_tides", nearest=1)
print(omnisea.coverage(multi).to_string(index=False))
print("\nEvery site you asked for gets a row, including the ones that found nothing.")
print("With a long list of locations, the gaps are the result you most need to see.")

joined = omnisea.to_dataframe(multi)
print("\nAnd the tidy frame carries your own site labels, ready to join:")
print(joined.groupby("site")["value"].agg(["count", "mean"]).round(3).to_string())


# ---------------------------------------------------------------------------
rule("10. Save it")

tree.to_netcdf("bamfield.nc")
print("wrote bamfield.nc — reopen losslessly with:")
print("    import xarray as xr; xr.open_datatree('bamfield.nc')")

import xarray as xr  # noqa: E402  (kept here to keep the example self-contained)

reopened = xr.open_datatree("bamfield.nc")
print(f"\ngroups round-tripped: {len([n for n in reopened.subtree if n.dataset.data_vars])}")
print("\nDone.")
