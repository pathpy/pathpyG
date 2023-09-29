from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Any, Optional
from functools import singledispatch
from collections import defaultdict, Counter

import operator
import numpy as np
import torch
import torch_geometric.utils
from scipy.sparse import linalg as spl
from scipy.sparse import csgraph

from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph
# from pathpyG.core.PathData import PathData


def temporal_graph_to_event_dag(g: TemporalGraph, delta=np.infty) -> Graph:
    """For a given temporal network, create a directed acyclic event graph where nodes are
        node-time events and edges between events capture time-respecting paths.
    """
    edge_list = []
    node_names = {}
    edge_times = []

    for v, w, t in g.temporal_edges:

        if delta < np.infty:
            current_delta = int(delta)
        else:
            current_delta = g.end_time - g.start_time

        event_src = "{0}-{1}".format(v, t)
        node_names[event_src] = v

        # create one time-unfolded link for all delta in [1, delta]
        # this implies that for delta = 2 and an edge (a,b,1) two
        # time-unfolded links (a_1, b_2) and (a_1, b_3) will be created
        for x in range(1, int(current_delta)+1):
            event_dst = "{0}-{1}".format(w, t+x)
            node_names[event_dst] = w
            edge_list.append([event_src, event_dst])
            edge_times.append(t)

    dag = Graph.from_edge_list(edge_list)
    dag.data['node_name'] = [ node_names[dag.node_index_to_id[v]] for v in range(dag.N) ]
    dag.data['node_idx'] = [ g.node_id_to_index[v] for v in dag.data['node_name']]
    dag.data['edge_ts'] = torch.tensor(edge_times)
    return dag


# def dfs_paths(dag: Graph, node) -> PathData:
#     src = []
#     dst = []

#     visited = set()
#     stack = [ node ]

#     while stack:
#         x = stack.pop()
#         for w in dag.successors(x):
#             if w not in visited:
#                 visited.add(w)
#                 stack.append(w)
#                 src.append(dag.node_id_to_index[x])
#                 dst.append(dag.node_id_to_index[w])
#     return PathData()

# def event_dag_all_subpaths(dag: Graph, l=1, mapping={}) -> PathData:
#     """For a given directed acyclic event graph, calculate all walks up to length l
#     that are contained in walks starting at all root events."""

#     # process subtrees for all root events with in-degree zero
#     d = dag.degrees(mode='in')
#     for v in d:
#         if d[v] == 0:
#             print('Processing causal root', v)

#             ctr = defaultdict(Counter)

#             # get edge_index of subtree reachable from root v
#             sub_tree_index = extract_tree(dag, v)

#             # get sparse matrix representation of DAG
#             A = torch_geometric.utils.to_scipy_sparse_matrix(sub_tree_index)
#             # paths of length zerp = nodes
#             for i in dag.nodes:
#                 ctr[0][(dag.node_id_to_index[i],)] += 1

#             # count all paths from length 1 to l
#             for k in range(1, l+1):
#                 for p in ctr[k-1]:
#                     for j in range(len(A.col)):
#                         if p[-1] == A.row[j]:
#                             ctr[k][p + (A.col[j],)] += 1

#             print(map_ctr(dag, ctr))
#     return PathData()

# def map_ctr(dag: Graph, ctr: defaultdict) -> defaultdict:
#     mapped_ctr = defaultdict(Counter)
#     for k in ctr:
#         for p in ctr[k]:
#             mapped_ctr[k][tuple([dag['node_name', dag.node_index_to_id[v]] for v in p])] += ctr[k][p]
#     return mapped_ctr



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
            causal_trees[v] = torch.IntTensor([src, dst])
    return causal_trees
