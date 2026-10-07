"""Export the marimo tutorials in `docs/tutorial/` to the Jupyter notebooks rendered in the documentation.

The marimo notebooks (`*.py`) are the source of the tutorials. Since executing them takes a long time,
the exported Jupyter notebooks (`*.ipynb`), including the outputs of all cells, are committed to the
repository and rendered by the `mkdocs-jupyter` plugin without executing them again.

Usage:

    # Re-export (and execute) all notebooks whose `.ipynb` is out of sync with its `.py`
    uv run --extra cpu python docs/export_tutorials.py

    # Re-export specific notebooks, e.g. after a change to pathpyG that affects their outputs
    uv run --extra cpu python docs/export_tutorials.py docs/tutorial/basic/basic_concepts.py

    # Only check that all `.ipynb` are in sync with their `.py` (exits with status 1 otherwise)
    uv run python docs/export_tutorials.py --check
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TUTORIAL_DIR = Path("docs", "tutorial")
MAX_WORKERS = 4
# Share the CPU cores between the notebooks exported in parallel. Otherwise, each notebook uses all cores for torch
# operations, and the resulting oversubscription slows down the export drastically
THREADS_PER_WORKER = str(max(1, (os.cpu_count() or 1) // MAX_WORKERS))


def export(notebook: Path, output: Path, execute: bool) -> None:
    """Export a marimo notebook to a Jupyter notebook, optionally including the outputs of all cells."""
    cmd = [sys.executable, "-m", "marimo", "export", "ipynb", str(notebook), "-o", str(output), "--sort", "top-down"]
    if execute:
        cmd.append("--include-outputs")
    env = {"OMP_NUM_THREADS": THREADS_PER_WORKER, "MKL_NUM_THREADS": THREADS_PER_WORKER} | os.environ
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Exporting {notebook} failed:\n{result.stdout}\n{result.stderr}")


def cell_sources(ipynb: Path) -> list[tuple[str, str]]:
    """Return the type and source of all cells of a Jupyter notebook, ignoring their outputs."""
    cells = json.loads(ipynb.read_text("utf-8"))["cells"]
    return [(cell["cell_type"], "".join(cell["source"])) for cell in cells]


def is_in_sync(notebook: Path) -> bool:
    """Check whether the committed `.ipynb` of a marimo notebook contains the same cells as the notebook itself."""
    ipynb = notebook.with_suffix(".ipynb")
    if not ipynb.exists():
        return False
    with tempfile.TemporaryDirectory() as tmp_dir:
        exported = Path(tmp_dir, ipynb.name)
        export(notebook, exported, execute=False)
        return cell_sources(exported) == cell_sources(ipynb)


def update(notebook: Path) -> None:
    """Execute a marimo notebook and export it, including all outputs, to the `.ipynb` next to it."""
    print(f"Exporting {notebook}", flush=True)
    export(notebook, notebook.with_suffix(".ipynb"), execute=True)
    print(f"Finished exporting {notebook}", flush=True)


def main() -> int:
    """Re-export or check the tutorial notebooks and return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "notebooks", nargs="*", type=Path, help="marimo notebooks to re-export (default: all out of sync)"
    )
    parser.add_argument("--check", action="store_true", help="only check that all notebooks are in sync")
    args = parser.parse_args()

    # Notebooks prefixed with `_` (e.g. in `archive/`) are not part of the documentation
    notebooks = args.notebooks or [
        nb for nb in sorted(TUTORIAL_DIR.rglob("*.py")) if not any(p.startswith("_") for p in nb.parts)
    ]
    # Explicitly given notebooks are always re-exported, all others only if they are out of sync
    if args.check or not args.notebooks:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            notebooks = [nb for nb, ok in zip(notebooks, executor.map(is_in_sync, notebooks)) if not ok]
    if args.check:
        for notebook in notebooks:
            print(f"{notebook.with_suffix('.ipynb')} is out of sync with {notebook}")
        if notebooks:
            print("Run `uv run --extra cpu python docs/export_tutorials.py` to re-export them")
        return 1 if notebooks else 0

    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for notebook, future in [(nb, executor.submit(update, nb)) for nb in notebooks]:
            try:
                future.result()
            except RuntimeError as e:
                print(e, file=sys.stderr)
                failed.append(notebook)
    if failed:
        print(f"Failed to export: {', '.join(map(str, failed))}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
