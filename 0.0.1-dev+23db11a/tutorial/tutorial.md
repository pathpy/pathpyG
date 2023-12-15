# Overview

In this tutorial, we will introduce basic concepts of pathpyG. pathpyG can be used as a wrapper around pytorch-geometric that facilitates network analysis, graph learning, and interactive data visualization. However, its real power comes into play when modelling causal path structures in time series data on networks, such as trajectories on graphs or temporal graphs with time-stamped interactions. pathpyG allows to compute causal paths in temporal graphs and model them based on [higher-order De Bruijn graphs](https://doi.org/10.1145/3097983.3098145), a higher-dimensional generalization of standard graph models for relational data.

The following introductory video explains the basic idea of higher-order De Bruiujn graph models for causal path structures in time series data:


<style>
/* https://github.com/squidfunk/mkdocs-material/issues/492 */

.video-wrapper {
    position: relative;
    display: block;
    height: 0;
    padding: 0;
    overflow: hidden;
    padding-bottom: 56.25%;
  }
  .video-wrapper > iframe {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border: 0;
  }
</style>

<div class="video-wrapper">
<iframe width="1280" height="720" src="https://www.youtube.com/embed/CxJkVrD2ZlM" title="When is a network a network?" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

The science behind pathpyG has been published in outlets like SIGKDD, WWW, Learning on Graphs, Nature Communications, Nature Physics, and Physical Review Letters. Please [check here](about.md) for more details on key scientific works that have laid the foundations for this package.

Different from previous versions of pathpy, the latest version pathpyG fully utilizes the power of torch and tensor-based representations of sparse graph models to failitate the use of higher-order De Bruijn graph models. pathpyG's data structures naturally generalize the concepts of [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/), which makes it easy to apply it in (temnporal) graph learning tasks.

Finally, pathpyG comes with an implementation of [De Bruijn Graph Neural Networks (DBGNN)](https://proceedings.mlr.press/v198/qarkaxhija22a.html), a causality-aware deep learning architecture for temporal graph data. In the tutorial, we illustrate this temporal graph learning approach in a simple toy example.

