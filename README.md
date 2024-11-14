[testing-image]: https://github.com/pathpy/pathpyG/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/pathpy/pathpyG/actions/workflows/testing.yml
[linting-image]: https://github.com/pathpy/pathpyG/actions/workflows/linting.yml/badge.svg
[linting-url]: https://github.com/pathpy/pathpyG/actions/workflows/linting.yml


pathpyG
=======

![image](docs/img/pathpy_logo_new.png)

[![Testing Status][testing-image]][testing-url]
[![Linting Status][linting-image]][linting-url]

pathpyG provides methods for GPU-accelerated Next-Generation Network Analytics and Graph Learning for Time Series Data on Temporal Networks.

The foundation of pathpyG are recent research results on the modelling of causal structures in temporal graph data based on higher-order De Bruijn graph models which generalize commonly used graph models. This perspective has been developed at ETH Zürich, University of Zürich, Princeton University and Julius-Maximilians-Universität Würzburg. Recent works include: 

- F Heeg, I Scholtes: [Using Time-Aware Graph Neural Networks to Predict Temporal Centralities in Dynamic Graphs](https://arxiv.org/abs/2310.15865), NeurIPS 2024, December 2024
- L Qarkaxhija, V Perri, I Scholtes: [De Bruijn goes Neural: Causality-Aware Graph Neural Networks for Time Series Data on Dynamic Graphs](https://proceedings.mlr.press/v198/qarkaxhija22a.html), Proceedings of the First Learning on Graphs Conference, PMLR 198:51:1-51:21, December 2022
- L Petrovic, I Scholtes: [Learning the Markov order of paths in graphs](https://doi.org/10.1145/3485447.3512091), Proceedings of WWW '22: The Web Conference 2022, Lyon, France, April 2022
- V Perri, I Scholtes: [HOTVis: Higher-Order Time-Aware Visualisation of Dynamic Graphs](https://doi.org/10.1007/978-3-030-68766-3_8), Proceedings of the 28th International Symposium on Graph Drawing and Network Visualization (GD 2020), Vancouver, BC, Canada, September 15-18, 2020
- R Lambiotte, M Rosvall, I Scholtes: [From Networks to Optimal Higher-Order Models of Complex Systems](https://www.nature.com/articles/s41567-019-0459-y), Nature Physics, Vol. 15, p. 313-320, March 25 2019
- I Scholtes: [When is a network a network? Multi-Order Graphical Model Selection in Pathways and Temporal Networks](http://dl.acm.org/citation.cfm?id=3098145), KDD'17 - Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Halifax, Nova Scotia, Canada, August 13-17, 2017


Documentation
-------------

Online documentation is available at [pathpy.net](https://www.pathpy.net).

The documentation includes multiple tutorials that introduce the use of pathpyG to model temporal graph and path data. You will also find an API reference and other useful information that will help you to get started.


Dependencies
------------

pathpyG supports Python 3.10+.

Installation requires [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), [torch](hhttps://pytorch.org/), and [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/).


Installation
------------

The latest development version can be installed from Github as follows:

    pip install git+https://github.com/pathpy/pathpyg.git


Testing
-------

To test pathpy, run `pytest` in the root directory.

This will exercise both the unit tests and docstring examples (using `pytest`).


Development
-----------

pathpyG development takes place on Github: https://github.com/pathpy/pathpyG

Please submit any reproducible bugs you encounter to the [issue tracker](https://github.com/pathpy/pathpyG/issues).
