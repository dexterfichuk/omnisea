"""Every registered source must satisfy the contract in docs/adding-a-provider.md.

This is the check a contributor runs before opening a pull request, and the one that keeps the
built-in sources honest. It found two real defects and one library-wide bug the day it was
written, so it earns its place.
"""

from __future__ import annotations

import pytest

import omnisea
from omnisea.conformance import (
    Problem,
    cf_standard_names,
    check_all,
    check_source,
    format_report,
)
from omnisea.providers.eccc import EcccClimateHourly, EcccProvider


def test_every_registered_source_conforms():
    problems = check_all()
    errors = [p for p in problems if p.level == "error"]
    assert not errors, "\n" + format_report(problems)


def test_no_warnings_either():
    """Warnings are things a reviewer would ask about; the built-ins should have none."""
    problems = check_all()
    assert not problems, "\n" + format_report(problems)


def test_the_bundled_cf_table_looks_like_the_real_thing():
    names = cf_standard_names()
    assert len(names) > 5000
    assert "sea_water_temperature" in names
    assert "sea_surface_wave_significant_height" in names
    # The name that reads as the obvious one for a tide gauge and is not in the table.
    assert "sea_surface_height_above_reference_datum" not in names


class TestTheCheckerItself:
    """A checker that cannot fail a bad source is worse than none."""

    def _broken(self, **overrides):
        source = EcccClimateHourly(EcccProvider())
        for key, value in overrides.items():
            setattr(source, key, value)
        return source

    def test_it_rejects_an_invented_standard_name(self):
        from omnisea import cf

        source = self._broken(
            fields={"X": cf.FieldSpec(var="x", standard_name="not_a_real_cf_name", units="m")}
        )
        assert any("not in the CF standard name table" in p.message for p in check_source(source))

    def test_it_rejects_two_fields_colliding_on_one_variable(self):
        from omnisea import cf

        source = self._broken(
            fields={
                "A": cf.FieldSpec(var="dup", standard_name="air_temperature", units="degC"),
                "B": cf.FieldSpec(var="dup", standard_name="air_temperature", units="degC"),
            },
            equivalent_fields=(),
        )
        assert any("both map to" in p.message for p in check_source(source))

    def test_a_declared_equivalence_makes_that_collision_legitimate(self):
        from omnisea import cf

        source = self._broken(
            fields={
                "A": cf.FieldSpec(var="dup", standard_name="air_temperature", units="degC"),
                "B": cf.FieldSpec(var="dup", standard_name="air_temperature", units="degC"),
            },
            equivalent_fields=(frozenset({"A", "B"}),),
        )
        assert not any("both map to" in p.message for p in check_source(source))

    def test_it_rejects_a_variable_name_that_collides_with_qc_naming(self):
        from omnisea import cf

        source = self._broken(fields={"X": cf.FieldSpec(var="thing_qc", standard_name="")})
        assert any("QC flag naming" in p.message for p in check_source(source))

    def test_it_rejects_a_conversion_with_nothing_to_convert_to(self):
        from omnisea import cf

        source = self._broken(
            fields={"X": cf.FieldSpec(var="x", standard_name="air_temperature",
                                      units="degC", cf_offset=273.15)}
        )
        assert any("no cf_units" in p.message for p in check_source(source))

    def test_it_rejects_a_bad_period_alias(self):
        assert any("Period alias" in p.message for p in check_source(self._broken(period="MS")))

    def test_it_rejects_a_non_timedelta_retention(self):
        assert any("pd.Timedelta" in p.message for p in check_source(self._broken(retention=30)))

    def test_it_rejects_an_unknown_feature_type(self):
        source = self._broken(feature_type="spreadsheet")
        assert any("CF DSG type" in p.message for p in check_source(source))

    def test_a_problem_prints_readably(self):
        assert "ERROR" in str(Problem("error", "src", "boom"))

    def test_the_report_is_encouraging_when_clean(self):
        assert format_report([]) == "All sources conform."

    def test_it_is_exported_for_contributors(self):
        assert omnisea.check_source is check_source
        assert omnisea.check_all is check_all


@pytest.mark.network
def test_the_bundled_cf_table_matches_the_published_one():
    """Guards against the bundled copy drifting from CF."""
    import re

    import requests

    url = (
        "https://cfconventions.org/Data/cf-standard-names/current/src/"
        "cf-standard-name-table.xml"
    )
    published = set(re.findall(r'<entry id="([^"]+)"', requests.get(url, timeout=180).text))
    bundled = cf_standard_names()
    missing = published - bundled
    assert not missing, (
        f"{len(missing)} names published but not bundled, e.g. {sorted(missing)[:5]}"
    )


class TestChecksFoundMissingByAContributor:
    """A contributor built a real provider from the docs and got past the checker with each
    of these. Every one produces a silently wrong result rather than a loud failure."""

    def source(self, **overrides):
        from omnisea import cf
        from omnisea.providers.base import Provider, RetrievalSource

        class P(Provider):
            name, title, license, base_url = "t", "T", "CC0", "https://example.org"

            def build_sources(self):
                return []

        attrs = {
            "name": "t_daily", "title": "T", "node_path": "in_situ/t", "period": "D",
            "fields": {
                "daily_rain_mm": cf.FieldSpec(
                    var="precipitation_amount", standard_name="precipitation_amount", units="mm"
                )
            },
            "discover": lambda self, q: [],
            "fetch": lambda self, q, m: [],
            **overrides,
        }
        return type("S", (RetrievalSource,), attrs)(P())

    def test_an_interval_statistic_named_the_other_way_round_is_caught(self):
        """`total_rain_mm` was caught by the prefix rule; `daily_rain_mm` — the field the
        shipped template teaches this with — was not. align() then interpolates a daily total
        into an intra-day distribution nobody measured."""
        problems = check_source(self.source())
        assert any("cell_methods" in p.message for p in problems)

    def test_declaring_the_cell_methods_clears_it(self):
        from omnisea import cf

        fixed = {
            "daily_rain_mm": cf.FieldSpec(
                var="precipitation_amount", standard_name="precipitation_amount",
                units="mm", cell_methods="time: sum",
            )
        }
        assert not [p for p in check_source(self.source(fields=fixed))
                    if "cell_methods" in p.message]

    def test_a_property_of_an_extreme_is_not_mistaken_for_one(self):
        """"period of the maximum wave" is not itself a maximum — taking its max when
        resampling would report a period no wave actually had."""
        from omnisea import cf

        fields = {
            "pd_of_max_wave_hgt": cf.FieldSpec(
                var="sea_surface_wave_period_of_highest_wave",
                standard_name="sea_surface_wave_period_of_highest_wave", units="s",
            )
        }
        assert not [p for p in check_source(self.source(fields=fields, period=None))
                    if "cell_methods" in p.message]

    @pytest.mark.parametrize("bad", [
        "/in_situ/leading", "in_situ/trailing/", "in_situ/has spaces",
        "in_situ//doubled", "../escape",
    ])
    def test_a_node_path_that_is_not_a_valid_netcdf_group_is_caught(self, bad):
        """Each segment becomes a netCDF group name; these all produced real files."""
        problems = check_source(self.source(node_path=bad))
        assert any("node_path" in p.message for p in problems), bad

    def test_an_ordinary_node_path_is_accepted(self):
        assert not [p for p in check_source(self.source(node_path="in_situ/my_network"))
                    if "node_path" in p.message]
