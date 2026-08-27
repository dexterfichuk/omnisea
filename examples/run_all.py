"""Re-execute every example notebook against the live APIs, in place.

The notebooks are committed with their outputs, because a notebook without them documents
nothing — every number in the prose is meant to be the number the cell above it actually
printed. That only stays true if they are re-run when the library changes, so this is the one
command that does it::

    python examples/run_all.py                 # all of them
    python examples/run_all.py prince_rupert   # just the ones whose name matches

It is deliberately *not* a test. It hits a dozen public APIs belonging to other people and takes
minutes; ``tests/test_examples.py`` checks the committed result instead, offline.

Exits non-zero if any notebook fails, naming the cell that broke.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(patterns: list[str]) -> int:
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError
    except ImportError:
        print('re-running notebooks needs the examples extra: pip install -e ".[examples]"')
        return 2

    notebooks = sorted(
        path
        for path in HERE.glob("*.ipynb")
        if not patterns or any(p in path.name for p in patterns)
    )
    if not notebooks:
        print(f"no notebooks matched {patterns or ['*']} in {HERE}")
        return 1

    failed: list[str] = []
    for path in notebooks:
        print(f"--- {path.name} ", end="", flush=True)
        started = time.monotonic()
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook, timeout=900, kernel_name="python3",
            resources={"metadata": {"path": str(HERE)}},
        )
        try:
            client.execute()
        except CellExecutionError as exc:
            # Write what did run: a half-executed notebook is what you need to debug it.
            nbformat.write(notebook, path)
            print(f"FAILED after {time.monotonic() - started:.0f}s\n{exc}")
            failed.append(path.name)
            continue
        nbformat.write(notebook, path)
        print(f"ok ({time.monotonic() - started:.0f}s)")

    if failed:
        print(f"\n{len(failed)} of {len(notebooks)} failed: {', '.join(failed)}")
        return 1
    print(f"\nall {len(notebooks)} notebooks re-executed cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
