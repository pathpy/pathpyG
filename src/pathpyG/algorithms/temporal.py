from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Any, Optional
from functools import singledispatch
from collections import defaultdict

import operator
import numpy as np
import torch
import torch_geometric.utils
from scipy.sparse import linalg as spl
from scipy.sparse import csgraph

from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph


def temporal_graph_to_dag(g: TemporalGraph, delta=np.infty):

    edge_list = []

    node_names = {}
    edge_times = []

    for v, w, t in g.temporal_edges:

        if delta < np.infty:
            current_delta = int(delta)
        else:
            current_delta = g.end_time - g.start_time

        v_t = "{0}-{1}".format(v, t)
        node_names[v_t] = v

        # create one time-unfolded link for all delta in [1, delta]
        # this implies that for delta = 2 and an edge (a,b,1) two
        # time-unfolded links (a_1, b_2) and (a_1, b_3) will be created
        for x in range(1, int(current_delta)+1):
            w_t = "{0}-{1}".format(w, t+x)
            node_names[w_t] = w
            edge_list.append([v_t, w_t])
            edge_times.append(t)

    g = Graph.from_edge_list(edge_list)
    g.data['node_name'] = [ node_names[g.node_index_to_id[v]] for v in range(g.N) ]
    g.data['edge_ts'] = torch.tensor(edge_times)
    return g