[testing-image]: https://github.com/pathpy/pathpyG/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/pathpy/pathpyG/actions/workflows/testing.yml
[linting-image]: https://github.com/pathpy/pathpyG/actions/workflows/linting.yml/badge.svg
[linting-url]: https://github.com/pathpy/pathpyG/actions/workflows/linting.yml


pathpyG
=======

[![Testing Status][testing-image]][testing-url]
[![Linting Status][linting-image]][linting-url]

GPU-accelerated Next-Generation Network Analytics and Graph Learning for Time Series Data on Dynamic Networks.

Documentation
-------------

Online documentation is available at [pathpy.net](https://www.pathpy.net).

The docs include a tutorials, an API reference, and other useful information.


Dependencies
------------

pathpyG supports Python 3.10+.

Installation requires [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), [torch](hhttps://pytorch.org/), and [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/).


Installation
------------

The development version can be installed from Github as follows:

    pip install git+https://github.com/pathpy/pathpyg.git


Testing
-------

To test pathpy, run `pytest` in the root directory.

This will exercise both the unit tests and docstring examples (using `pytest`).


Development
-----------

pathpyG development takes place on Github: https://github.com/pathpy/pathpyG

Please submit any reproducible bugs you encounter to the [issue tracker](https://github.com/pathpy/pathpyG/issues).
