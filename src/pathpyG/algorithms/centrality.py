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
from tqdm import tqdm

from collections import defaultdict, Counter, deque
from pathpyG.algorithms.temporal import temporal_shortest_paths, lift_order_temporal
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


def betweenness_centrality(g: Graph, sources=None):
    """
    Calculates the betweenness centrality of nodes based on the algorithm 
    proposed by Brandes.
    """
    bw = defaultdict(lambda: 0.0)

    if sources == None:
        sources = [v for v in g.nodes]

    for s in sources:
        S = list()
        P = defaultdict(list)

        sigma = defaultdict(lambda: 0)  
        sigma[s] = 1

        d = defaultdict(lambda: -1)        
        d[s] = 0

        Q = [s]
        while Q:
            v = Q.pop(0)
            S.append(v)
            for w in g.successors(v):
                if d[w] < 0:
                    Q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    # we found shortest path from s via v to w
                    sigma[w] = sigma[w] + sigma[v]
                    P[w].append(v)
        delta = defaultdict(lambda: 0.0)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] = delta[v] + sigma[v]/sigma[w] * (1 + delta[w])
                if v != w:
                    bw[w] = bw[w] + delta[w]
    return bw


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


# def fo_node(v, g, src_indices) -> int:
#     # if v is one of the source nodes, return corresponding first-order node
#     if v in src_indices:
#         return v - g.M
#     else: # return first-order target node otherwise
#         return g.data.edge_index[1,v].int().item()


def temporal_betweenness_centrality(g: TemporalGraph, delta=1):

    # generate temporal event DAG
    edge_index = lift_order_temporal(g, delta)

    # Add indices of first-order nodes as src of paths in augmented
    # temporal event DAG
    src_edges_src = g.data.edge_index[0] + g.M
    src_edges_dst = torch.arange(0, g.data.edge_index.size(1))


    # add edges from source to edges and from edges to destinations
    src_edges = torch.stack([src_edges_src, src_edges_dst])
    edge_index = torch.cat([edge_index, src_edges], dim=1)

    event_graph = Graph.from_edge_index(edge_index, num_nodes=g.M+g.N)
    #print(event_graph)

    # sources = first-order source nodes in temporal event graph
    src_indices = torch.unique(src_edges_src).tolist()

    e_i = g.data.edge_index.numpy()    

    fo_nodes = dict()
    for v in range(g.M+g.N):
        if v < g.M: # return first-order target node otherwise
            fo_nodes[v] = e_i[1,v]
        else:
            fo_nodes[v] = v - g.M

    bw = defaultdict(lambda: 0.0)

    # for all first-order nodes
    for s in tqdm(src_indices):

        # for any given s, d[v] is the shortest path distance from s to v
        # Note that here we calculate topological distances from sources to events (i.e. time-stamped edges)
        delta_ = defaultdict(lambda: 0.0)

        # for any given s, sigma[v] counts shortest paths from s to v
        sigma = defaultdict(lambda: 0.0)
        sigma[s] = 1

        sigma_fo = defaultdict(lambda: 0.0)
        sigma_fo[fo_nodes[s]] = 1

        dist = defaultdict(lambda: -1)
        dist[s] = 0

        dist_fo = defaultdict(lambda: -1)
        dist_fo[fo_nodes[s]] = 0
                
        # for any given s, P[v] is the set of predecessors of v on shortest paths from s
        P = defaultdict(set)

        # Q is a queue, so we append at the end and pop from the start
        Q = deque()
        Q.append(s)

        # S is a stack, so we append at the end and pop from the end
        S = list()
    
        # dijkstra with path counting
        while Q:
            v = Q.popleft()
            # for all successor events within delta
            for w in event_graph.successors(v):

                # we dicover w for the first time
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    if dist_fo[fo_nodes[w]] == -1:
                        dist_fo[fo_nodes[w]] = dist[v] + 1
                    S.append(w)
                    Q.append(w)
                # we found a shortest path to event w via event v
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[w] + sigma[v]
                    P[w].add(v)
                    # we found a shortest path to first-order node of event w
                    if dist[w] == dist_fo[fo_nodes[w]]:
                        sigma_fo[fo_nodes[w]] += sigma[v]
        
        c = 0
        for i in dist_fo:
            if dist_fo[i] >=0:
                c+= 1
        bw[fo_nodes[s]] = bw[fo_nodes[s]] - c + 1

        while S:
            w = S.pop()
            # work backwards through paths to all targets and sum delta and sigma   
            if dist[w] == dist_fo[fo_nodes[w]]:
                # v_fo = fo_tgt(v, g, src_indices, tgt_indices)
                delta_[w] += (sigma[w]/sigma_fo[fo_nodes[w]])
            for v in P[w]:
                delta_[v] += (sigma[v]/sigma[w]) * delta_[w]
                bw[fo_nodes[v]] += delta_[w] * (sigma[v]/sigma[w])
    bw_id = defaultdict(lambda: 0.0)
    for idx in bw:
        bw_id[g.mapping.to_id(idx)] = bw[idx]
    return bw_id


def temporal_closeness_centrality(g: TemporalGraph, delta: int) -> dict:
    """Calculates the closeness of nodes based on observed shortest paths
    between all nodes. Following the definition by M. A. Beauchamp 1965
    (https://doi.org/10.1002/bs.3830100205).

    Parameters
    ----------
    g: TemporalGraph
    delta: int

    Returns
    -------
    dict
    """
    centralities = dict()
    dist, _ = temporal_shortest_paths(g, delta)
    for x in g.nodes:
        centralities[x] = sum((g.N - 1) / dist[_np.arange(g.N) != g.mapping.to_idx(x), g.mapping.to_idx(x)])

    return centralities


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
