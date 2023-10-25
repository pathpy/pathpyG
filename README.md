[testing-image]: https://github.com/pathpy/pathpyG/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/pathpy/pathpyG/actions/workflows/testing.yml
[linting-image]: https://github.com/pathpy/pathpyG/actions/workflows/linting.yml/badge.svg
[linting-url]: https://github.com/pathpy/pathpyG/actions/workflows/linting.yml


pathpyG
=======

[![Testing Status][testing-image]][testing-url]
[![Linting Status][linting-image]][linting-url]

An Open Source package providing higher-order analytics and learning for time series data on graphs.

Documentation
-------------

Online documentation is available at [pathpy.net](https://www.pathpy.net).

The docs include a [tutorial](https://www.pathpy.net/tutorial.html), [example gallery](https://www.pathpy.net/examples/index.html), [API reference](https://www.pathpy.net/api.html), and other useful information.


Dependencies
------------

pathpyG supports Python 3.7+.

Installation requires [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), [torch](hhttps://pytorch.org/), and [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/).


Installation
------------

The development version can be installed from Github:

    pip install git+https://github.com/pathpy/pathpyg.git


Testing
-------

To test pathpy, run `make test` in the source directory.

This will exercise both the unit tests and docstring examples (using `pytest`).


Development
-----------

pathpyG development takes place on Github: https://github.com/pathpy/pathpyG

Please submit any reproducible bugs you encounter to the [issue tracker](https://github.com/pathpy/pathpyG/issues).
