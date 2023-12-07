# About

## What is pathpyG?

pathpyG is an Open Source package facilitating next-generation network analytics and graph learning for time series data on graphs.

pathpyG is tailored to analyse time-stamped network data as well as sequential data that capture multiple short paths observed in a graph or network. Examples for data that can be analysed with pathpyG include high-resolution time-stamped network data, dynamic social networks, user click streams on the Web, biological pathway data, citation graphs, passenger trajectories in transportation networks, or information propagation in social networks.

pathpyG is fully integrated with jupyter, providing rich interactive visualisations of networks, temporal networks, higher-, and multi-order models. Visualisations can be exported to HTML5 files that can be shared and published on the Web.

## What is the science behind pathpyG?

The theoretical foundation of this package, higher- and multi-order network models, was developed in the following peer-reviewed research articles:

1. R Lambiotte, M Rosvall, I Scholtes: [From networks to optimal models of complex systems](https://www.nature.com/articles/s41567-019-0459-y), Nature Physics 15, 313-320, March 2019
2. I Scholtes: [When is a network a network? Multi-Order Graphical Model Selection in Pathways and Temporal Networks](http://dl.acm.org/citation.cfm?id=3098145), In KDD'17 - Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Halifax, Nova Scotia, Canada, August 13-17, 2017
3. I Scholtes, N Wider, A Garas: [Higher-Order Aggregate Networks in the Analysis of Temporal Networks: Path structures and centralities](https://link.springer.com/article/10.1140/epjb/e2016-60663-0), The European Physical Journal B, 89:61, March 2016
4. I Scholtes, N Wider, R Pfitzner, A Garas, CJ Tessone, F Schweitzer: [Causality-driven slow-down and speed-up of diffusion in non-Markovian temporal networks](https://www.nature.com/articles/ncomms6024), Nature Communications, 5, September 2014
5. R Pfitzner, I Scholtes, A Garas, CJ Tessone, F Schweitzer: [Betweenness preference: Quantifying correlations in the topological dynamics of temporal networks](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.198701), Phys Rev Lett, 110(19), 198701, May 2013

A broader view on the importance of higher-order network models in network analysis can be found in [this article](https://arxiv.org/abs/1806.05977). An explanatory video with a high-level introduction of the the science behind pathpyG is available below. 

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
