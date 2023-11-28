from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
)

from pathpyG.core.Graph import Graph

import torch_geometric
from networkx import centrality

def map_to_nodes(g: Graph, c: Dict):
    if len(g.node_index_to_id) > 0:
        return {g.node_index_to_id[i]: c[i] for i in c}
    return c

def __getattr__(name: str) -> Any:
    """Map unknown methods to corresponding method of networkx."""
    def wrapper(*args, **kwargs) -> Any:
        # print('unknown method {0} was called, delegating call to networkx centralities'.format(name))
        g = torch_geometric.utils.to_networkx(args[0].data)
        return map_to_nodes(args[0], getattr(centrality, name)(*args, **kwargs))
    return wrapper
