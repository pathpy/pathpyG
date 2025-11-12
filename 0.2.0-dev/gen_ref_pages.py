"""Generate the code reference pages and navigation."""
# See for more detail: https://mkdocstrings.github.io/recipes/

from pathlib import Path

import mkdocs_gen_files
import yaml

nav = mkdocs_gen_files.Nav()
# Load the ignored modules from the YAML file
ignored_modules_path = Path("docs", "reference", "ignored_modules.yaml")
ignored_modules = yaml.safe_load(ignored_modules_path.read_text("utf-8"))

for path in sorted(Path("src").rglob("*.py")):
    if ignored_modules and str(path.relative_to(".")) in ignored_modules:
        print(f"Skipping {path} as it is in the ignored modules list.")
        continue
    module_path = path.relative_to("src").with_suffix("")
    doc_path = path.relative_to("src").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    parts_list = []
    for part in parts:
        if part.startswith("_"):
            parts_list.append(part.split("_")[-1])
        else:
            parts_list.append(part)
    
    nav[tuple(parts_list)] = doc_path.as_posix()

    print(f"Checking {full_doc_path}")
    if not (Path("docs") / full_doc_path).exists():
        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: {ident}")
    else:
        print(f"File {full_doc_path} already exists, skipping.")

    mkdocs_gen_files.set_edit_path(full_doc_path, Path("../") / path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
