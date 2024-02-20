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

from pathpyG import Graph
from pathpyG import TemporalGraph

from torch_geometric.utils import to_networkx
from networkx import centrality

from collections import defaultdict
import numpy as _np
from torch import tensor

def path_node_traversals(paths):
    """Calculate the number of times any path traverses each of the nodes.

    Parameters
    ----------
    paths: Paths

    Returns
    -------
    dict
    """
    traversals = defaultdict(lambda: 0)
    for path_id, path_edgelist in paths.paths.items():
        path_seq = paths.walk_to_node_seq(path_edgelist)
        for node in path_seq:
            traversals[node.item()] += paths.path_freq[path_id]
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

    return c


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

def shortest_paths(paths):
    """
    Calculates all shortest paths between all pairs of nodes 
    based on a set of empirically observed paths.
    """
    s_p = defaultdict(lambda: defaultdict(set))
    s_p_lengths = defaultdict(lambda: defaultdict(lambda: _np.inf))

    p_length = 1
    index, edge_weights = paths.edge_index_k_weighted(k=p_length)
    sources = index[0]
    destinations = index[-1]
    for e, (s, d) in enumerate(zip(sources, destinations)):
        s = s.item()
        d = d.item()
        s_p_lengths[s][d] = p_length
        s_p[s][d] = set({tensor([s, d])})
    p_length += 1
    while True: # until max path length
        try:
            index, edge_weights = paths.edge_index_k_weighted(k=p_length)
            sources = index[0, :, 0]
            destinations = index[1, :, -1]
            for e, (s, d) in enumerate(zip(sources, destinations)):
                s = s.item()
                d = d.item()
                if p_length < s_p_lengths[s][d]:
                    # update shortest path length
                    s_p_lengths[s][d] = p_length
                    # redefine set
                    s_p[s][d] = {paths.walk_to_node_seq(index[:, e])}
                elif p_length == s_p_lengths[s][d]:
                    s_p[s][d].add(paths.walk_to_node_seq(index[:, e]))
            p_length += 1
        except IndexError:
            # print(f"IndexError occurred. Reached maximum path length of {p_length}")
            break
    return s_p


def path_betweenness_centrality(paths, normalized=False):
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

    # Log.add('Calculating betweenness in paths ...', Severity.INFO)

    all_paths = shortest_paths(paths)

    for s in all_paths:
        for d in all_paths[s]:
            for p in all_paths[s][d]:
                for x in p[1:-1]:
                    if s != d != x:
                        node_centralities[x.item()] += 1.0 / len(all_paths[s][d])
    if normalized:
        max_centr = max(node_centralities.values())
        for v in node_centralities:
            node_centralities[v] /= max_centr
    # assign zero values to nodes not occurring on shortest paths
    nodes = [v.item() for v in paths.edge_index.reshape(-1).unique(dim=0)]
    for v in nodes:
        node_centralities[v] += 0
    # Log.add('finished.')
    return node_centralities


def path_distance_matrix(paths):
    """
    Calculates shortest path distances between all pairs of
    nodes based on the observed shortest paths (and subpaths)
    """
    dist = defaultdict(lambda: defaultdict(lambda: _np.inf))
    # Log.add('Calculating distance matrix based on empirical paths ...', Severity.INFO)
    nodes = [v.item() for v in paths.edge_index.reshape(-1).unique(dim=0)] # NOTE: modify once set of nodes can be obtained from path obeject
    for v in nodes:
        dist[v][v] = 0

    p_length = 1
    index, edge_weights = paths.edge_index_k_weighted(k=p_length)
    sources = index[0]
    destinations = index[-1]
    for e, (s, d) in enumerate(zip(sources, destinations)):
        s = s.item()
        d = d.item()
        dist[s][d] = p_length
        # s_p[s][d] = set({torch.tensor([s,d])})
    p_length += 1
    while True: # until max path length
        try:
            index, edge_weights = paths.edge_index_k_weighted(k=p_length)
            sources = index[0, :, 0]
            destinations = index[1, :, -1]
            for e, (s, d) in enumerate(zip(sources, destinations)):
                s = s.item()
                d = d.item()
                if p_length < dist[s][d]:
                    # update shortest path length
                    dist[s][d] = p_length
            p_length += 1
        except IndexError:
            #print(f"IndexError occurred. Reached maximum path length of {p_length}")
            break
    return dist


def path_closeness_centrality(paths, normalized=False):
    """Calculates the closeness of nodes based on observed shortest paths
    between all nodes

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
    distances = path_distance_matrix(paths)
    nodes = [v.item() for v in paths.edge_index.reshape(-1).unique(dim=0)] # NOTE: modify once set of nodes can be obtained from path obeject

    for x in nodes:
        # calculate closeness centrality of x
        for d in nodes:
            if x != d and distances[d][x] < _np.inf:
                node_centralities[x] += 1.0 / distances[d][x]

    # assign zero values to nodes not occurring
    
    for v in nodes:
        node_centralities[v] += 0.0

    if normalized:
        m = max(node_centralities.values())
        for v in nodes:
            node_centralities[v] /= m

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
            raise RuntimeError(f'Did not find method {name} with no arguments')
        if isinstance(args[0], TemporalGraph):
            raise NotImplementedError(f'Missing implementation of {name} for temporal graphs')
        # if first argument is of type Graph, delegate to networkx function    
        if isinstance(args[0], Graph):
            g = to_networkx(args[0].data)
            r = getattr(centrality, name)(g, *args[1:], **kwargs)
            if name.index('centrality') > 0 and isinstance(r, dict):
                return map_to_nodes(args[0], r)
            return r
        else:
            return wrapper(*args, **kwargs)
            #raise RuntimeError(f'Did not find method {name} that accepts first argument of type {type(args[0])}')
    return wrapper
