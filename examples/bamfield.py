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

import numpy as np
import pandas as pd

import omnisea

# Third-party deprecations only. omnisea's own warnings are left on: they are how it tells you
# a join is about to be wrong, and an example that hides them teaches the wrong habit.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# A Site is the location type: coordinates, a search radius, and the name you know it by.
# That name travels with the results, so what comes back is joinable to your own records.
BAMFIELD = omnisea.Site(48.8353, -125.1358, "Bamfield Marine Sciences Centre", radius_km=30)
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

catalog = omnisea.discover(sites=BAMFIELD, time=WEEK)

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

# Site objects, (lat, lon, name) tuples, dicts and DataFrames are all accepted, so a CSV
# of moorings you already keep works without reshaping it first.
sites = [
    omnisea.Site(48.8353, -125.1358, "Bamfield", radius_km=15),
    omnisea.Site(49.1530, -125.9066, "Tofino", radius_km=15),
    omnisea.Site(48.4204, -123.3656, "Victoria", radius_km=15),
    omnisea.Site(47.5000, -128.5000, "Open ocean", radius_km=15),  # deliberately empty
]

multi = omnisea.positions(sites, time=WEEK, providers="dfo_tides", nearest=1)
print(omnisea.coverage(multi).to_string(index=False))
print("\nEvery site you asked for gets a row, including the ones that found nothing.")
print("With a long list of locations, the gaps are the result you most need to see.")

joined = omnisea.to_dataframe(multi)
print("\nAnd the tidy frame carries your own site labels, ready to join:")
print(joined.groupby("site")["value"].agg(["count", "mean"]).round(3).to_string())


# ---------------------------------------------------------------------------
rule("10. Bring your own data and build a model")

rng = np.random.default_rng(0)
# Pacific local time, made explicit. Drop the tz_localize and omnisea will warn you that it is
# reading these as UTC -- which on a tide series is metres of error.
field_times = pd.to_datetime([
    "2024-07-01 09:14", "2024-07-01 15:40", "2024-07-02 10:05", "2024-07-03 08:50",
    "2024-07-04 14:20", "2024-07-05 09:00", "2024-07-06 16:30", "2024-07-07 11:10",
]).tz_localize("America/Vancouver")
mine = pd.DataFrame({
    "time": field_times,
    "chlorophyll_ug_L": rng.uniform(1.5, 8.0, len(field_times)).round(2),
})
print("your field sheet — irregular sampling times:")
print(mine.to_string(index=False))

# align(on=...) joins the environmental data onto YOUR timestamps and carries your columns
# through, so you get y and X in one table.
features = omnisea.align(tree, on=mine, tolerance="30min")
show = ["chlorophyll_ug_L", "water_surface_height_above_reference_datum",
        "air_temperature", "air_temperature_max", "precipitation_amount"]
print("\njoined to environmental predictors:")
print(features[show].round(3).to_string())

print("\nhow each column got there — chosen from CF cell_methods, not guessed:")
for column, how in features.attrs["omnisea_aggregation"].items():
    if "reviewed" not in column:
        print(f"  {column:56s} {how}")

print("\nA daily total matches by INTERVAL CONTAINMENT (the day holding your sample);")
print("an instantaneous tide height matches to the nearest reading within tolerance.")
print("Without that distinction a 30-minute tolerance against a midnight stamp")
print("would silently hand you a column of NaN.")

print("\nOr resample everything onto a regular grid instead:")
weekly = omnisea.align(tree, freq="7D")
print(f"  weekly precipitation total : {weekly['precipitation_amount'].sum():.1f} mm  (summed)")
print(f"  weekly max daily-max temp  : {weekly['air_temperature_max'].max():.1f} degC"
      "  (max, not mean)")

# and your own measurements can live in the tree beside the providers'
with_mine = omnisea.add_local(
    tree, mine, name="Chlorophyll grab samples",
    lat=48.8353, lon=-125.1358, station_id="BAM-CHL",
    var_attrs={"chlorophyll_ug_L": {"long_name": "Chlorophyll a", "units": "ug L-1"}},
)
print("\nyour data as a node in the tree:")
print(omnisea.summary(with_mine)[["node", "station_name", "n_time"]].to_string(index=False))


# ---------------------------------------------------------------------------
rule("11. Fit a model")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
except ImportError:
    print("scikit-learn is not installed; skipping "
          '(pip install -e ".[examples]" to run this section)')
else:
    # A mock logger deployment, sampling every two hours in Pacific local time. The response
    # is SYNTHETIC -- built from the real fetched drivers with coefficients we choose, plus
    # noise -- so this tests the pipeline, not oceanography.
    #
    # HOW it is built matters. We go around align() entirely: raw node series, interpolated
    # with plain pandas, and the daily lighthouse value looked up by LOCAL calendar date. If
    # we generated the response from align()'s output and then refit on align()'s output, any
    # systematic join error would appear identically on both sides and cancel exactly -- the
    # recovery would look perfect while the join was hours out. Building the truth outside the
    # join is what makes the recovery below an actual test of it.
    local = pd.date_range("2024-07-02 00:00", "2024-07-07 00:00", freq="2h",
                          tz="America/Vancouver")
    naive = local.tz_convert("UTC").tz_localize(None)

    tide_raw = tides["water_surface_height_above_reference_datum"].to_series()
    tide_at = (tide_raw.reindex(tide_raw.index.union(naive))
               .interpolate(method="time").reindex(naive).to_numpy())

    daily = weather["air_temperature_max"].to_series()
    # The lighthouse record is labelled by local calendar date, so a sample takes the value
    # for the local day it fell on. Looked up directly here, without asking align().
    tmax_at = daily.reindex(
        pd.DatetimeIndex(local.normalize().tz_localize(None))
    ).to_numpy()

    TRUE = {
        "intercept": 6.00,
        "water_surface_height_above_reference_datum": -0.45,
        "air_temperature_max": 0.30,
        "hour_sin": 0.60,
    }
    diurnal = np.sin(2 * np.pi * (local.hour.to_numpy() - 9) / 24)
    logger = pd.DataFrame({
        "time": local,
        "water_temp_c": (
            TRUE["intercept"]
            + TRUE["water_surface_height_above_reference_datum"] * tide_at
            + TRUE["air_temperature_max"] * tmax_at
            + TRUE["hour_sin"] * diurnal
            + rng.normal(0, 0.15, len(local))
        ).round(2),
    }).dropna()

    # ... and only now does align() see the problem, from the timestamps alone.
    data = omnisea.align(tree, on=logger, tolerance="30min")
    # align() hands back UTC; the diurnal term is a local-clock effect, so convert back.
    local_hours = data.index.tz_localize("UTC").tz_convert("America/Vancouver").hour
    data["hour_sin"] = np.sin(2 * np.pi * (local_hours - 9) / 24)

    predictors = ["water_surface_height_above_reference_datum",
                  "air_temperature_max", "hour_sin"]
    usable = data[[*predictors, "water_temp_c"]].dropna()
    X, y = usable[predictors], usable["water_temp_c"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    model = LinearRegression().fit(X_train, y_train)
    predicted = model.predict(X_test)

    print(f"{len(usable)} synthetic samples over 5 days, joined to real drivers")
    print(f"held-out R2  = {r2_score(y_test, predicted):.3f}")
    print(f"held-out MAE = {mean_absolute_error(y_test, predicted):.3f} degC")

    print("\ndid it recover the relationship we put in?")
    print(f"  {'term':44s} {'fitted':>8s} {'true':>7s}")
    print(f"  {'intercept':44s} {model.intercept_:8.3f} {TRUE['intercept']:7.2f}")
    for term, coefficient in zip(predictors, model.coef_, strict=True):
        print(f"  {term:44s} {coefficient:8.3f} {TRUE[term]:7.2f}")
    print("\nThe truth was constructed without touching align(), so those numbers agreeing")
    print("is evidence about the JOIN: a join off by the station's UTC offset, a daily total")
    print("that had been interpolated, or a unit silently converted would all show up here")
    print("as coefficients that miss.")

    # -- correlated features across the sources: see them, then drop them -------------
    print("\nMultiple sources publish the same physical signal several ways over, and a")
    print("model matrix built from all of them is collinear. See the redundancy first:")
    redundancy = omnisea.correlations(data, threshold=0.9)
    print(redundancy.to_string(index=False))

    pruned = omnisea.drop_correlated(data, threshold=0.9)
    print("\ndrop_correlated() keeps the better-covered column of each pair, and says why:")
    for column, why in pruned.attrs["omnisea_dropped"].items():
        print(f"  dropped {column:40s} {why}")
    print("\nYour own columns (here water_temp_c) are carried through align(on=...) and are")
    print("never dropped — correlation with YOUR data is the point of the model, not")
    print("redundancy. Pin any feature you trust with keep=.")

    def wide_fit(matrix: pd.DataFrame) -> tuple[int, float]:
        columns = [c for c in matrix.columns
                   if c != "water_temp_c"
                   and pd.api.types.is_numeric_dtype(matrix[c])
                   and matrix[c].nunique() > 1]
        features = matrix[columns].dropna(axis=1)
        train_x, test_x, train_y, test_y = train_test_split(
            features, matrix["water_temp_c"], test_size=0.3, random_state=0)
        fitted = LinearRegression().fit(train_x, train_y)
        return features.shape[1], r2_score(test_y, fitted.predict(test_x))

    n_all, r2_all = wide_fit(data)
    n_pruned, r2_pruned = wide_fit(pruned)
    print(f"\n  everything omnisea returned : {n_all:2d} predictors -> R2 = {r2_all:.3f}")
    print(f"  after drop_correlated       : {n_pruned:2d} predictors -> R2 = {r2_pruned:.3f}")
    print("\nSame predictive power from fewer, non-duplicated features — and every removal")
    print("is recorded in attrs['omnisea_dropped'], auditable like the resampling choices.")


# ---------------------------------------------------------------------------
rule("12. Save it")

tree.to_netcdf("bamfield.nc")
print("wrote bamfield.nc — reopen losslessly with:")
print("    import xarray as xr; xr.open_datatree('bamfield.nc')")

import xarray as xr  # noqa: E402  (kept here to keep the example self-contained)

reopened = xr.open_datatree("bamfield.nc")
print(f"\ngroups round-tripped: {len([n for n in reopened.subtree if n.dataset.data_vars])}")
print("\nDone.")
