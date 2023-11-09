# Contributing

This project is open source and welcomes contributions. In the following sections, you will find information about how to contribute to this project, set up your environment correctly, how to document your code and more.

## Overview

- [Contributing](#contributing)
  - [Overview](#overview)
  - [Setting up your environment](#setting-up-your-environment)
  - [Git pre-commit hooks](#git-pre-commit-hooks)
  - [Documentation](#documentation)
  - [Code Style](#code-style)
  - [Formatting](#formatting)
  - [Testing](#testing)
  - [Code of Conduct](#code-of-conduct)

## Setting up your environment

Dev containers bla bla

## Git pre-commit hooks

If you are wondering why every commit you make takes so long, it is because we run a couple of checks on your code before it is committed. These checks are configured as pre-commit hooks and are run automatically when you commit your code. The checks are documented in detail in `pre-commit-config.yaml`.  
They are installed by default via the Dev Container setup. If you want to install them manually, you can do so by running the following command after you installed the project `[dev]` dependencies:
```bash
pre-commit install
```

## Documentation

This project uses `mkdocs` for documentation. The documentation is hosted on GitHub Pages. The necessary `html`-files are built automatically from the `markdown`-files and `Jupyter`-notebooks in the `docs/`-directory and the `Python`-files in `src/`. You can host the documentation locally with the following command:
```bash
mkdocs serve
```
The documentation is then available at `http://localhost:8000/`.

The `Code Reference` is generated automatically from the source files. The docstrings should be formatted according to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). Be sure to also use the advanced stuff like notes, tips and more. They can e.g. look as follows:

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

!!! todo
    Add more information about the documentation and short examples on how to do it.

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
    1. This runs `pyright` on all files in the current directory. You can also run `pyright` on a single file or a subdirectory by specifying the path accordingly.

## Formatting

We use `black` for formatting. You can run it locally with the following command:

```bash
black . # (1)!
```

1. Watch out! This command will format all files in the current directory. You can also run `black` on a single file or a subdirectory by specifying the path accordingly.
2.
We further use `isort` for sorting imports. You can run it locally with the following command:
```bash
isort .
```
The default keyboard shortcut for formatting in `VSCode` is `Alt + Shift + F`.

## Testing

Pytest blabla

## Code of Conduct

Be nice!
