"""CF canonicalization: encoding repairs, opt-in unit conversion, and passthrough."""

from __future__ import annotations

import pytest

from omnisea import cf
from omnisea.providers.eccc import EcccClimateDaily, EcccClimateHourly, EcccProvider


@pytest.fixture
def hourly():
    return EcccClimateHourly(EcccProvider())


@pytest.fixture
def daily():
    return EcccClimateDaily(EcccProvider())


class TestEncodingRepairs:
    def test_wind_direction_tens_of_degrees_is_multiplied(self, hourly):
        """ECCC's `25` means 250 degrees. This is an encoding, not a unit choice."""
        spec = hourly.fields["WIND_DIRECTION"]
        assert cf.convert(25, spec) == 250.0
        assert cf.convert(36, spec) == 360.0

    def test_gust_direction_gets_the_same_repair(self, daily):
        spec = daily.fields["DIRECTION_MAX_GUST"]
        assert cf.convert(8, spec) == 80.0

    def test_repair_applies_even_without_cf_units(self, hourly):
        """The x10 fix is unconditional — it is not part of the opt-in conversion."""
        spec = hourly.fields["WIND_DIRECTION"]
        assert cf.convert(25, spec, to_cf_units=False) == 250.0
        assert cf.convert(25, spec, to_cf_units=True) == 250.0

    def test_missing_values_stay_missing(self, hourly):
        """None must not become 273.15."""
        assert cf.convert(None, hourly.fields["TEMP"], to_cf_units=True) is None


class TestUnitConversion:
    def test_values_stay_in_provider_units_by_default(self, hourly):
        assert cf.convert(15.0, hourly.fields["TEMP"]) == 15.0

    def test_opt_in_conversion_reaches_canonical_cf_units(self, hourly):
        assert cf.convert(15.0, hourly.fields["TEMP"], to_cf_units=True) == pytest.approx(288.15)

    def test_kpa_to_pa(self, hourly):
        got = cf.convert(101.12, hourly.fields["STATION_PRESSURE"], to_cf_units=True)
        assert got == pytest.approx(101120.0)

    def test_kmh_to_ms(self, hourly):
        got = cf.convert(18.0, hourly.fields["WIND_SPEED"], to_cf_units=True)
        assert got == pytest.approx(5.0)

    def test_units_attribute_always_describes_the_actual_values(self, hourly):
        spec = hourly.fields["TEMP"]
        assert cf.cf_attrs(spec)["units"] == "degC"
        assert cf.cf_attrs(spec, to_cf_units=True)["units"] == "K"

    def test_unconverted_variable_advertises_how_to_convert(self, hourly):
        attrs = cf.cf_attrs(hourly.fields["TEMP"])
        assert attrs["cf_units"] == "K"
        assert "to_cf_units=True" in attrs["note"]


class TestStandardNames:
    def test_quantity_without_a_cf_name_emits_no_standard_name(self, hourly):
        """There is no CF standard name for humidex; inventing one would be worse than none."""
        attrs = cf.cf_attrs(hourly.fields["HUMIDEX"])
        assert "standard_name" not in attrs
        assert attrs["long_name"] == "Humidex"

    def test_daily_temperature_statistics_share_a_standard_name_via_cell_methods(self, daily):
        mean, low, high = (
            daily.fields["MEAN_TEMPERATURE"],
            daily.fields["MIN_TEMPERATURE"],
            daily.fields["MAX_TEMPERATURE"],
        )
        assert mean.standard_name == low.standard_name == high.standard_name == "air_temperature"
        assert {mean.var, low.var, high.var} == {
            "air_temperature",
            "air_temperature_min",
            "air_temperature_max",
        }
        assert low.cell_methods == "time: minimum"

    def test_plan_era_alias_resolves_to_the_real_cf_name(self):
        """`sea_surface_height_above_reference_datum` is not in the CF table."""
        resolved = cf.resolve_names(["sea_surface_height_above_reference_datum"])
        assert "water_surface_height_above_reference_datum" in resolved


class TestPassthrough:
    def test_unmapped_fields_are_carried_not_dropped(self, hourly):
        specs = cf.resolve_fields(
            hourly.fields, ["TEMP", "WEATHER_ENG_DESC"], skip=hourly.effective_skip()
        )
        assert specs["WEATHER_ENG_DESC"].var == "WEATHER_ENG_DESC"
        assert specs["TEMP"].var == "air_temperature"

    def test_passthrough_is_tagged_so_it_can_be_told_apart(self, hourly):
        specs = cf.resolve_fields(hourly.fields, ["WEATHER_ENG_DESC"])
        attrs = cf.cf_attrs(specs["WEATHER_ENG_DESC"])
        assert attrs[cf.MAPPED_ATTR] == 0
        assert attrs["source_field"] == "WEATHER_ENG_DESC"

    def test_mapped_fields_are_tagged_too(self, hourly):
        assert cf.cf_attrs(hourly.fields["TEMP"])[cf.MAPPED_ATTR] == 1

    def test_passthrough_can_be_switched_off(self, hourly):
        specs = cf.resolve_fields(
            hourly.fields, ["TEMP", "WEATHER_ENG_DESC"], include_unmapped=False
        )
        assert set(specs) == {"TEMP"}

    def test_qc_siblings_do_not_become_variables(self, hourly):
        specs = cf.resolve_fields(
            hourly.fields, ["TEMP", "TEMP_FLAG"], is_qc=hourly.is_qc_field
        )
        assert "TEMP_FLAG" not in specs

    def test_illegal_characters_are_made_netcdf_safe(self):
        spec = cf.passthrough_spec("avg wind-speed (10m)")
        assert spec.var == "avg_wind_speed_10m"
        assert spec.long_name == "avg wind-speed (10m)"

    def test_colliding_names_are_disambiguated_never_merged(self):
        table = {
            "a": cf.FieldSpec(var="dup", standard_name=""),
            "b": cf.FieldSpec(var="dup", standard_name=""),
        }
        specs = cf.resolve_fields(table, ["a", "b"])
        assert {s.var for s in specs.values()} == {"dup", "dup_2"}


class TestVariableSelection:
    def test_request_by_cf_standard_name(self, hourly):
        specs = cf.resolve_fields(hourly.fields, ["TEMP", "WIND_SPEED"], requested=["wind_speed"])
        assert set(specs) == {"WIND_SPEED"}

    def test_request_by_raw_provider_field_name(self, hourly):
        specs = cf.resolve_fields(hourly.fields, ["TEMP", "WIND_SPEED"], requested=["TEMP"])
        assert set(specs) == {"TEMP"}

    def test_request_by_standard_name_matches_all_its_statistics(self, daily):
        specs = cf.resolve_fields(
            daily.fields,
            ["MEAN_TEMPERATURE", "MIN_TEMPERATURE", "TOTAL_SNOW"],
            requested=["air_temperature"],
            include_unmapped=False,
        )
        assert set(specs) == {"MEAN_TEMPERATURE", "MIN_TEMPERATURE"}


class TestEovBridge:
    def test_goos_eov_maps_to_cf_names(self):
        assert cf.eov_to_cf(["seaSurfaceTemperature"]) == ("sea_surface_temperature",)

    def test_currents_expand_to_both_components(self):
        assert len(cf.eov_to_cf(["surfaceCurrents"])) == 2

    def test_eov_without_a_cf_equivalent_is_dropped_not_faked(self):
        assert cf.eov_to_cf(["fishAbundanceAndDistribution"]) == ()

    def test_unknown_eov_is_ignored(self):
        assert cf.eov_to_cf(["notAnEov"]) == ()
