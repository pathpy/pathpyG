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
from pathpyG.core.PathData import PathData


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
    # dag.data['node_name'] = [ node_names[g.node_index_to_id[v]] for v in range(g.N) ]
    dag.data['edge_ts'] = torch.tensor(edge_times)
    return dag


def event_dag_all_walks(dag: Graph, l=1) -> PathData:
        """For a given directed acyclic event graph, calculate all subwalks up to length l
        that are contained in the paths starting in all root events."""
        path_data = PathData()

        print(dag.data.edge_index)

        # process subtrees for all root events with in-degree zero
        d = dag.degrees(mode='in')
        for v in d:
            if d[v] == 0:
                print('Processing causal root', v)
                # TODO: get edge_index of tree reachable from root v

                ctr = defaultdict(Counter)

                sub_tree_index = torch.tensor([[0, 1, 1],
                                              [1, 2, 3]])

                # get sparse matrix representation of DAG
                A = torch_geometric.utils.to_scipy_sparse_matrix(sub_tree_index)
                print(A)

                for i in A.row:
                    ctr[0][(i,)] += 1

                # perform l sparse matrix multiplications of matrix A with itself
                for k in range(1, l+1):
                    for p in ctr[l-1]:
                        for j in range(len(A.col)):
                            if p[-1] == A.row[j]:
                                ctr[k][p + (A.col[j],)] += 1

                print(ctr)
        return path_data
