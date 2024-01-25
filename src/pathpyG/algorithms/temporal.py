"""Algorithms for the analysis of causal path structures in temporal graphs."""


from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Union, List
from collections import defaultdict

import numpy as np
import torch

from pathpyG import Graph
from pathpyG import TemporalGraph
from pathpyG import config


def temporal_graph_to_event_dag(g: TemporalGraph, delta: float = np.infty,
                                sparsify: bool = True) -> Graph:
    """Create directed acyclic event graph where nodes are node-time events and edges are time-respecting paths.

        Args:
            g: the temporal graph to be convered to an event dag
            delta: for a maximum time difference delta two consecutive interactions 
                    (u, v; t) and (v, w; t') are considered to contribute to a causal
                    path from u via v to w iff t'-t <= delta
            sparsify: whether not to add edges for time-respecting paths
                between same nodes for multiple inter-event times
    """
    edge_list = []
    node_names = {}
    edge_times = []

    sources = defaultdict(set)

    for v, _w, t in g.temporal_edges:
        sources[t].add(v)

    for v, w, t in g.temporal_edges:

        if delta < np.infty:
            current_delta = int(delta)
        else:
            current_delta = g.end_time - g.start_time

        event_src = f"{v}-{t}"
        node_names[event_src] = v

        # create one time-unfolded link for all delta in [1, delta]
        # this implies that for delta = 2 and an edge (a,b,1) two
        # time-unfolded links (a_1, b_2) and (a_1, b_3) will be created
        cont = False
        for x in range(1, int(current_delta)+1):

            # only add edge to event DAG if an edge (w,*) continues a time-respecting path at time t+x
            if w in sources[t+x] or not sparsify:
                event_dst = "{0}-{1}".format(w, t+x)
                node_names[event_dst] = w
                edge_list.append([event_src, event_dst])
                edge_times.append(t)
                cont = True
        if not cont and sparsify:  # if there is no continuing time-respecting path, include edge to t+1
            event_dst = "{0}-{1}".format(w, t+1)
            node_names[event_dst] = w
            edge_list.append([event_src, event_dst])
            edge_times.append(t)

    dag = Graph.from_edge_list(edge_list)
    dag.data['node_name'] = [node_names[dag.mapping.to_id(i)] for i in range(dag.N)]
    dag.data['node_idx'] = [g.mapping.to_idx(v) for v in dag.data['node_name']]
    dag.data['edge_ts'] = torch.tensor(edge_times)
    return dag


def extract_causal_trees(dag: Graph) -> Dict[Union[int, str], torch.IntTensor]:
    """Extract all causally isolated trees from a directed acyclic event graph.

    For a directed acyclic graph where all events are related to single root
    event, this function will return a single tree. For other DAGs, it will return
    multiple trees such that each root in the tree is not causally influenced by
    any other node-time event.

    Args:
        dag: the event graph to process
    """
    causal_trees = {}
    d = dag.degrees(mode='in')
    for v in d:
        if d[v] == 0:
            # print('Processing root', v)

            src: List[int] = []
            dst: List[int] = []

            visited = set()
            queue = [v]

            while queue:
                x = queue.pop()
                for w in dag.successors(x):
                    if w not in visited:
                        visited.add(w)
                        queue.append(w)
                        src.append(dag.mapping.to_idx(x))
                        dst.append(dag.mapping.to_idx(w))
            # TODO: Remove redundant zero-degree neighbors of all nodes
            causal_trees[v] = torch.IntTensor([src, dst]).to(config['torch']['device'])
    return causal_trees
