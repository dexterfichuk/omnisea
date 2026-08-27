"""The example notebooks are documentation, so they are checked like documentation.

A notebook committed with a traceback in it, or with its outputs stripped, teaches whatever it
last did rather than what it claims to do — and it is the first thing a new user runs. These
checks are offline and read the committed JSON: they do not re-execute anything, they assert
that what was committed is the result of a clean run.

Re-executing them is a separate, deliberate act -- ``python examples/run_all.py`` -- because
it hits every live API these notebooks touch, and a test suite should not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))


def cells(path: Path) -> list[dict]:
    return json.loads(path.read_text())["cells"]


def code_cells(path: Path) -> list[dict]:
    return [c for c in cells(path) if c["cell_type"] == "code" and "".join(c["source"]).strip()]


def test_there_are_notebooks_to_check():
    """A glob that silently matches nothing would make every test below vacuously pass."""
    assert NOTEBOOKS, f"no notebooks found under {EXAMPLES}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
class TestACommittedNotebookIsACleanRun:
    def test_no_cell_errored(self, path):
        failures = []
        for i, cell in enumerate(cells(path)):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    failures.append(
                        f"cell {i}: {output.get('ename')}: {output.get('evalue')}"
                    )
        assert not failures, f"{path.name} was committed with tracebacks in it:\n" + "\n".join(
            failures
        )

    def test_every_code_cell_ran(self, path):
        """An unrun cell means the committed outputs came from a different notebook than this
        one — the numbers below it belong to code that is no longer there."""
        unrun = [
            i
            for i, cell in enumerate(cells(path))
            if cell["cell_type"] == "code"
            and "".join(cell["source"]).strip()
            and cell.get("execution_count") is None
        ]
        assert not unrun, f"{path.name}: code cells {unrun} have no execution count"

    def test_the_cells_ran_in_order(self, path):
        """Out-of-order counts mean the reader cannot reproduce it by running top to bottom,
        which is the only way anybody actually runs a notebook."""
        counts = [c["execution_count"] for c in code_cells(path)]
        assert counts == sorted(counts), f"{path.name} ran out of order: {counts}"

    def test_it_is_explained_and_not_just_executed(self, path):
        """Prose is the point. A notebook of bare cells is a script with extra steps."""
        markdown = [c for c in cells(path) if c["cell_type"] == "markdown"]
        code = code_cells(path)
        assert len(markdown) >= len(code) / 2, (
            f"{path.name}: {len(code)} code cells to {len(markdown)} markdown — "
            "most of it is unexplained"
        )

    def test_no_output_is_an_empty_promise(self, path):
        """A code cell that ran and printed nothing, in a notebook whose job is to show what
        came back, is usually a query that quietly found nothing."""
        silent = [
            i
            for i, cell in enumerate(cells(path))
            if cell["cell_type"] == "code"
            and "".join(cell["source"]).strip()
            and not cell.get("outputs")
        ]
        # Assignments and imports legitimately print nothing; a majority doing so does not.
        assert len(silent) <= len(code_cells(path)) * 0.7, (
            f"{path.name}: {len(silent)} of {len(code_cells(path))} code cells produced no "
            "output at all"
        )
