"""Algorithms to calculate centralities in (temporal) graphs.

The functions and submodules in this module allow to compute 
time-respecting or causal paths in temporal graphs and to
calculate (temporal) and higher-order graph metrics like centralities.

Example:
    ```py
    # Import pathpyG and configure your torch device if you want to use GPU acceleration.
    import pathpyG as pp
    pp.config['torch']['device'] = 'cuda'

    # Generate toy example for temporal graph
    g = pp.TemporalGraph.from_edge_list([
        ['b', 'c', 2],
        ['a', 'b', 1],
        ['c', 'd', 3],
        ['d', 'a', 4],
        ['b', 'd', 2],
        ['d', 'a', 6],
        ['a', 'b', 7]
    ])

    # Extract DAG capturing causal interaction sequences in temporal graph
    dag = pp.algorithms.temporal_graph_to_event_dag(g, delta=1)

    # Get path object to calculate statistics.
    paths = pp.PathData.from_temporal_dag(dag)

    # Generate weighted (first-order) time-aggregated graph
    g = pp.HigherOrderGraph(paths, order=1)

    # Call networkx function `closeness_centrality` on graph
    c = pp.algorithms.centrality.closeness_centrality(g)
    ```
"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
)

from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph
from pathpyG.core.DAGData import DAGData

from networkx import centrality

from collections import defaultdict, Counter
from pathpyG.algorithms.temporal import temporal_shortest_paths, time_respecting_paths
import numpy as _np
import torch
from torch import tensor

from torch_geometric.utils import to_networkx, degree


def path_node_traversals(dags: DAGData) -> Counter:
    """Calculate the number of times any dag traverses each of the nodes.

    Parameters
    ----------
    dags: DAGData

    Returns
    -------
    Counter
    """
    traversals = Counter()
    for dag in dags.dags:
        t = torch.maximum(
            degree(dag.edge_index[1], num_nodes=dag.num_nodes), degree(dag.edge_index[0], num_nodes=dag.num_nodes)
        )
        for v in range(len(t)):
            # TODO: Re-evaluate the weight representation
            traversals[dags.mapping.to_id(v)] += t[v].item() * dag.edge_weight.max().item()
    return traversals


def map_to_nodes(g: Graph, c: Dict) -> Dict:
    """Map node-level centralities in dictionary to node IDs.

    Args:
        g: Graph object
        c: dictionary mapping node indices to metrics

    Example:
        ```pycon
        >>> import pathpyG as pp
        >>> g = pp.Graph(torch.LongTensor([[1, 1, 2], [0, 2, 1]]),
        ...                               node_id=['a', 'b', 'c'])
        >>> c = {0: 0.5, 1: 2.7, 2: 0.3}
        >>> c_mapped = pp.algorithms.centrality.map_to_nodes(g, c)
        >>> print(c_mapped)
        {'a': 0.5, 'b': 2.7, 'c': 0.3}
        ```
    """
    return {g.mapping.to_id(i): c[i] for i in c}



def path_visitation_probabilities(paths):
    """Calculates the probabilities that a randomly chosen path passes through each of
    the nodes. If 5 out of 100 paths (of any length) traverse node v, node v will be
    assigned a visitation probability of 0.05. This measure can be interpreted as ground
    truth for the notion of importance captured by PageRank applied to a graphical
    abstraction of the paths.

    Parameters
    ----------
    paths: Paths

    Returns
    -------
    dict
    """
    # if not isinstance(paths, PathData):
    #    assert False, "`paths` must be an instance of Paths"
    # Log.add('Calculating visitation probabilities...', Severity.INFO)

    # entries capture the probability that a given node is visited on an arbitrary path
    # Note: this is identical to the subpath count of zero-length paths
    # (i.e. the relative frequencies of nodes across all pathways)
    visit_probabilities = path_node_traversals(paths)

    # total number of visits
    visits = 0.0
    for v in visit_probabilities:
        visits += visit_probabilities[v]

    for v in visit_probabilities:
        visit_probabilities[v] /= visits
    # Log.add('finished.', Severity.INFO)
    return visit_probabilities


def temporal_betweenness_centrality(g: TemporalGraph, delta: int) -> dict:
    """Calculates the betweenness of nodes based on observed shortest paths
    between all pairs of nodes

    Parameters
    ----------
    paths:
        Paths object
    normalized: bool
        normalize such that largest value is 1.0

    Returns
    -------
    dict
    """
    # assert isinstance(paths, pp.PathData), "argument must be an instance of pathpy.Paths"
    node_centralities = defaultdict(lambda: 0)

    sp, _, counts = temporal_shortest_paths(g, delta)

    for v in range(g.N):
        for paths in list(sp.values())[1:]:
            v_as_start = paths[:, 0] == v
            v_as_end = paths[:, -1] == v
            paths_not_v = paths[~(v_as_start | v_as_end)]
            mask = paths_not_v == v
            fractions = (mask.sum(dim=1) > 0) / counts[paths_not_v[:, 0], paths_not_v[:, -1]]
            node_centralities[g.mapping.to_id(v)] += fractions.sum().item()

    return node_centralities


def temporal_closeness_centrality(g: TemporalGraph, delta: int) -> dict:
    """Calculates the closeness of nodes based on observed shortest paths
    between all nodes. Following the definition by M. A. Beauchamp 1965
    (https://doi.org/10.1002/bs.3830100205).

    Parameters
    ----------
    paths: Paths
    normalized: bool
        normalize such that largest value is 1.0

    Returns
    -------
    dict
    """
    node_centralities = defaultdict(lambda: 0)
    _, sp_lengths, _ = temporal_shortest_paths(g, delta)
    sp_lengths.fill_diagonal_(float("inf"))
    print(sp_lengths)
    for v in g.nodes:
        print(f"v: {v}, idx: {g.mapping.to_idx(v)}")
        print((g.N - 1) / sp_lengths[:, g.mapping.to_idx(v)])
        node_centralities[v] = ((g.N - 1) / sp_lengths[:, g.mapping.to_idx(v)]).sum().item()

    return node_centralities


def __getattr__(name: str) -> Any:
    """Map to corresponding functions in centrality module of networkx.

    Any call to a function that is not implemented in the module centrality
    and whose first argument is of type Graph will be delegated to the
    corresponding function in the networkx module `centrality`. Please
    refer to the [networkx documentation](https://networkx.org/documentation/stable/reference/algorithms/centrality.html)
    for a reference of available functions.

    Args:
        name: the name of the function that shall be called
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if len(args) == 0:
            raise RuntimeError(f"Did not find method {name} with no arguments")
        if isinstance(args[0], TemporalGraph):
            raise NotImplementedError(f"Missing implementation of {name} for temporal graphs")
        # if first argument is of type Graph, delegate to networkx function
        if isinstance(args[0], Graph):
            g = to_networkx(args[0].data)
            r = getattr(centrality, name)(g, *args[1:], **kwargs)
            if name.index("centrality") > 0 and isinstance(r, dict):
                return map_to_nodes(args[0], r)
            return r
        else:
            return wrapper(*args, **kwargs)
            # raise RuntimeError(f'Did not find method {name} that accepts first argument of type {type(args[0])}')

    return wrapper
