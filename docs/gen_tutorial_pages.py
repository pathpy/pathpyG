"""Generate the tutorial pages from the marimo notebooks in `docs/tutorial/`.

Each marimo notebook is exported to a Jupyter notebook (including the outputs of all cells)
which is then rendered by the `mkdocs-jupyter` plugin.
"""

import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mkdocs_gen_files

TUTORIAL_DIR = Path("docs", "tutorial")
# Notebooks that are exported without executing them, since they take too long to run
NO_EXECUTE = {"manim_tutorial.py", "dbgnn.py"}


def export(notebook: Path) -> tuple[Path, str]:
    """Export a marimo notebook to a Jupyter notebook and return its contents."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ipynb = Path(tmp_dir, notebook.with_suffix(".ipynb").name)
        cmd = [sys.executable, "-m", "marimo", "export", "ipynb", str(notebook), "-o", str(ipynb), "--sort", "top-down"]
        if notebook.name not in NO_EXECUTE:
            cmd.append("--include-outputs")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Exporting {notebook} failed:\n{result.stdout}\n{result.stderr}")
        return notebook, ipynb.read_text("utf-8")


# Notebooks prefixed with `_` (e.g. in `archive/`) are not part of the documentation
notebooks = [nb for nb in sorted(TUTORIAL_DIR.rglob("*.py")) if not any(p.startswith("_") for p in nb.parts)]
with ThreadPoolExecutor(max_workers=4) as executor:
    for notebook, content in executor.map(export, notebooks):
        with mkdocs_gen_files.open(notebook.relative_to("docs").with_suffix(".ipynb"), "w") as f:
            f.write(content)
        mkdocs_gen_files.set_edit_path(notebook.relative_to("docs").with_suffix(".ipynb"), notebook.relative_to("docs"))
