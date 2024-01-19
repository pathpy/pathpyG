# Contributing

This project is open source and welcomes contributions. In the following sections, you will find information about how to contribute to this project, set up your environment correctly, how to document your code and more.

## Overview

- [Setting up your environment](#setting-up-your-environment)
- [Documentation](#documentation)
- [Code Style](#code-style)
- [Formatting](#formatting)
- [Testing](#testing)

## Setting up your environment

### Clone the Repository
The first step is to clone the repository. You can do this by running the following command:
```bash
git clone https://github.com/pathpy/pathpyG
```
If you do not have the rights to push to the repository, you can also fork the repository and clone your fork instead. From there you can create a pull request to the original repository.

### Installation

To ensure version consistency, we use a [Development Container](https://containers.dev/) for this project. :vscode_logo: VSCode provides an easy-to-use extension for this. Check out their [official documentation](https://code.visualstudio.com/docs/devcontainers/containers) for more information. Once you've installed the extension successfully, :vscode_logo: VSCode will recommend reopening the project in the Dev Container. You can also do this manually by clicking on the button in the bottom left corner of :vscode_logo: VSCode and then selecting `Reopen in Container`.

??? note "Setup without Dev Containers"
    If you do not want to use Dev Containers, you can also install the dependencies into your virtual Python environment manually. We recommend that you follow the instructions provided on our [getting started](getting_started.md) page. As last step, install the package in editable mode and include the dependencies necessary for testing, documentation and general development:
    ```bash
    pip install -e '.[dev,test,doc]'
    ```

### Git pre-commit hooks

If you are wondering why every commit you make takes so long, it is because we run a couple of checks on your code before it is committed. These checks are configured as pre-commit hooks and are running automatically when you commit your code. The checks are documented in detail in `pre-commit-config.yaml`.  
They are installed by default in the Dev Container setup. If you installed the package manually, you can set up the hooks by running the following command:
```bash
pre-commit install
```

## Documentation

This project uses [`MkDocs`](https://www.mkdocs.org/) for documentation. It is a static site generator that creates the necessary `html`-files automatically from the `markdown`-files and [:jupyter_logo: Jupyter](https://jupyter.org/) notebooks in the `docs/`-directory and the `Python`-files in `src/`. The documentation is hosted on GitHub Pages.

### Hosting the documentation locally

You can host the documentation locally with the following command:
```bash
mkdocs serve
```
The documentation is then available at [`http://localhost:8000/`](http://localhost:8000/).

??? info "Actual Deployment"
    The development version of the documentation is deployed automatically to GitHub Pages when something is pushed to the `main`-branch. The workflow for deploying a new stable version needs to be triggered manually. You can find it in the `Actions`-tab of the repository. Both workflows use [`mike`](https://github.com/jimporter/mike) instead of `MkDocs` to enable versioning.

### Code Reference

The `Code Reference` is generated automatically from the :python_logo: Python source files. The docstrings should be formatted according to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). Be sure to also use the advanced stuff like notes, tips and more. They can e.g. look as follows:

=== "Docstring"
    ```python
    """
    Note:
        This is a note.

    Tip: This is a heading
        This is a tip.
    """
    ```
=== "Result"
    !!! note

        This is a note.

    !!! tip "This is a heading"

        This is a tip.

See the documentation of the underlying [griffe](https://mkdocstrings.github.io/griffe/docstrings/) package for more details.

To get an overview for each module, `mkdocstrings` automatically uses the docstrings from the `__init__.py` files in each module as description. Thus, do not forget to add a docstring to each `__init__.py` file.

### Tutorials

The tutorials are written in :jupyter_logo: Jupyter notebooks. They are located in the `docs/`-directory. You can add new tutorials by adding the notebook to the `docs/tutorial/`-directory and adding the path to the `mkdocs.yml`-file under `nav:`. The tutorials are automatically converted to `html`-files when the documentation is built.

### Adding new pages

You can add more pages to the documentation by adding a `markdown`-file to the `docs/`-directory and adding the path to the `mkdocs.yml`-file under `nav:`. The pages are automatically converted to `html`-files when the documentation is built. We are using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) as a theme. It includes many great features like annotations, code blocks, diagrams, admonitions and more. Check out their [documentation](https://squidfunk.github.io/mkdocs-material/reference/) for more information.

## Code Style

We (soon) enforce code style guidelines with `pylint`, `flake8`, `mypy` and `pyright`. These packages are configured as defaults in the Dev Container setup via `VSCode` and the settings are saved in `pyproject.toml`. You can run them locally with the following commands:

- `pylint`: A linter that checks for errors and code style violations.
    ```bash
    pylint scr/ # (1)!
    ```
    1. This runs `pylint` on all files in `scr/`. You can also run `pylint` on a single file by specifying the path to the file instead.
- `flake8`: Another linter that checks for bad code smells and suspicious constructs.
    ```bash
    flake8 . # (1)!
    ```
    1. This runs `flake8` on all files in the current directory. You can also run `flake8` on a single file or a subdirectory by specifying the path accordingly.
- `mypy`: A static type checker for Python.
    ```bash
    mypy src/ # (1)!
    ```
    1. This runs `mypy` on all files in `src/`. You can also run `mypy` on a single file by specifying the path to the file instead.
- `pyright`: A second static type checker for Python.
    ```bash
    pyright . # (1)!
    ```
    1. This runs `pyright` on all files in the current directory. You can also run it on a single file or a subdirectory by specifying the path accordingly.

## Formatting

We use `black` for formatting. You can run it locally with the following command:

```bash
black . # (1)!
```

1. This command will format all files in the current directory. You can also run `black` on a single file or a subdirectory by specifying the path accordingly.

We further use `isort` for sorting imports. You can run it locally with the following command:
```bash
isort .
```
The default keyboard shortcut for formatting in `VSCode` is `Alt + Shift + F`.

## Testing

We are using `pytest` for testing. You can run the tests locally with the following command:
```bash
pytest
```
The tests are located in the `tests/`-directory. We use `pytest-cov` to measure the test coverage and are aiming for 100% coverage with a hard limit of 80%. Tests will fail if the coverage drops below 80%.

!!! todo "Add tests"
    We are currently only at 29% coverage. So the lines above are currently pure fiction.
