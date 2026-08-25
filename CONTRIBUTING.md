# Contributing to omnisea

The main thing omnisea wants from contributors is **more data sources**. Adding one is designed
to be a single new file and a pull request, not a negotiation about architecture.

## Adding a data source

```bash
git clone https://github.com/dexterfichuk/omnisea && cd omnisea
uv venv && uv pip install -e ".[dev]"
pytest -m "not network"        # should be green before you start
```

1. **Copy the template.** [`examples/provider_template.py`](examples/provider_template.py) is a
   complete, runnable source with every hook commented. Start there, not from scratch.
2. **Read [`docs/adding-a-provider.md`](docs/adding-a-provider.md).** It is the contract: the
   two base classes, `FieldSpec` semantics, and the traps that have already bitten us.
3. **Probe the API before you write the field table.** Every trap in this codebase was found by
   `curl`, not by reading documentation. The one that has cost us most: an endpoint's own
   metadata is often wrong. ECCC's station catalogue overstates several periods of record, and
   IWLS sends `Cache-Control: no-store` on a station list that changes once a year.
4. **Check your work:**

```bash
python -m omnisea.conformance      # the contract, as a program
pytest -m "not network"
ruff check src/ tests/
```

`omnisea.check_source(MySource(MyProvider()))` checks just yours while you iterate.

5. **Register it** in your provider's `build_sources()`, and open the PR.

## What a reviewer will look for

The conformance checker covers the mechanical rules. A human review is about the things it
cannot see:

- **Did you probe, or assume?** Say in the PR what you verified and how. "`hydrometric-daily-mean`
  returns 8 rows for station 08HB014 over 2024-07-01/08" is worth more than a paragraph of
  description.
- **Are the `cell_methods` right?** They are not decoration — `omnisea.align()` reads them to
  decide how a variable resamples. A daily total tagged as instantaneous gets interpolated,
  which invents an intra-day distribution that was never measured. AHCCD's `temp_max` turned out
  to be the *mean of daily maxima*, not a monthly extreme; tagged `time: maximum` it would have
  produced a maximum over a series of means.
- **What did you leave out, and why?** Omitting something is fine. Omitting it silently is not.
  `climate-normals` is deliberately unsupported, with a network test pinning the reason.
- **Does it fail loudly?** A source that returns `[]` when it should raise, or raises when it
  should return `[]`, is worse than one that does not exist. If your dataset is a rolling
  archive, declare `retention` so a historical query gets an explanation instead of silence.

## Tests

- **Unit tests over committed fixtures.** Capture small *real* responses into `tests/fixtures/`
  with a prefix of your own, and parse those. Do not hand-write JSON that looks like what you
  think the API returns — the point of a fixture is that it is what the API *actually* returned.
- **A few `@pytest.mark.network` tests** proving the thing works end to end. Keep them small and
  tolerant of a slow public server, but never write an assertion that passes vacuously.
- **Test the traps.** If you found a surprise, encode it. `test_a_row_with_no_date_at_all_still_lands_on_its_year`
  is a better test name than `test_annual_peaks`.

## Style

- Comments explain **why**, not what. If a line needs explaining, the explanation is usually a
  fact about the upstream API that is not visible in the code.
- Errors should be `omnisea` errors from `errors.py`, so `except omnisea.OmniseaError` stays a
  complete catch.
- `ruff check` and `ruff format` conventions are in `pyproject.toml`; line length is 100.

## Reporting a bug

Include the query that triggered it. A `Catalog` repr or `omnisea.citation(tree)` output is
ideal — both record exactly which sources answered and under what terms.

If omnisea returned data that was *wrong* rather than absent, say so prominently. Silent
wrongness is the failure mode this library exists to prevent, and those reports get priority.
