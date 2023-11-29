from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Union
from collections import defaultdict

import numpy as np
import torch
from scipy.sparse import linalg as spl

from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph
from pathpyG import config


def temporal_graph_to_event_dag(g: TemporalGraph, delta=np.infty, sparsify=True) -> Graph:
    """For a given temporal network, create a directed acyclic event graph where nodes are
        node-time events and edges between events capture time-respecting paths.
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

            # only add edge to event DAG if an edge (w,*) continues a time-repsecing path at time t+x
            if w in sources[t+x] or not sparsify:
                event_dst = "{0}-{1}".format(w, t+x)
                node_names[event_dst] = w
                edge_list.append([event_src, event_dst])
                edge_times.append(t)
                cont = True
        if not cont and sparsify: # if there is no continuing time-respecting path, include edge to t+1
            event_dst = "{0}-{1}".format(w, t+1)
            node_names[event_dst] = w
            edge_list.append([event_src, event_dst])
            edge_times.append(t)

    dag = Graph.from_edge_list(edge_list)
    dag.data['node_name'] = [ node_names[dag.node_index_to_id[v]] for v in range(dag.N) ]
    dag.data['node_idx'] = [ g.node_id_to_index[v] for v in dag.data['node_name']]
    dag.data['edge_ts'] = torch.tensor(edge_times)
    return dag

def extract_causal_trees(dag: Graph) -> Dict[Union[int, str], torch.IntTensor]:

    causal_trees = {}
    d = dag.degrees(mode='in')
    for v in d:
        if d[v] == 0:
            # print('Processing root', v)

            src = []
            dst = []

            visited = set()
            queue = [ v ]

            while queue:
                x = queue.pop()
                for w in dag.successors(x):
                    if w not in visited:
                        visited.add(w)
                        queue.append(w)
                        if len(dag.node_id_to_index)>0:
                            src.append(dag.node_id_to_index[x])
                            dst.append(dag.node_id_to_index[w])
                        else:
                            src.append(x)
                            dst.append(w)
            # TODO: Remove redundant zero-degree neighbors of all nodes
            causal_trees[v] = torch.IntTensor([src, dst]).to(config['torch']['device'])
    return causal_trees
