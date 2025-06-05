from __future__ import annotations
from typing import Any, Optional, Union

import ast
import re
import warnings

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data

from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.core.index_map import IndexMap

# Regex to check if the attribute is iterable (e.g., list, dict, etc.)
_iterable_re = re.compile(r"^\s*[\[\{\(].*[\]\}\)]\s*$")
_number_re = re.compile(
    r"""^\s*          # optional leading whitespace
    [+-]?                              # optional sign
    (                                  # start group
        (\d+\.\d*)|(\.\d+)|(\d+)       # float or int
    )
    ([eE][+-]?\d+)?                    # optional exponent
    \s*$                               # optional trailing whitespace
"""
)
_integer_re = re.compile(r"^\s*[+-]?\d+\s*$")


def _check_column_name(frame: pd.DataFrame, name: str, synonyms: list) -> pd.DataFrame:
    """Helper function to check column names and change them if needed."""
    if name not in frame.columns:
        for col in frame.columns:
            if col in synonyms:
                frame.rename(columns={col: name}, inplace=True)
                continue
    return frame


def _parse_df_column(df: pd.DataFrame, data: Data, attr: str, idx: list | None = None, prefix: str = "") -> None:
    """Helper function to parse a column in a DataFrame and add it as an attribute to the graph."""
    if idx is None:
        idx = np.arange(len(df))

    if df[attr].dtype == "object":
        if _iterable_re.match(str(df[attr].values[0])):
            data[prefix + attr] = torch.tensor(
                [ast.literal_eval(x) for x in df[attr].values[idx]], device=data.edge_index.device
            )
        elif _number_re.match(str(df[attr].values[0])):
            # if the attribute is a number, convert it to a tensor
            if _integer_re.match(str(df[attr].values[0])):
                data[prefix + attr] = torch.tensor(
                    df[attr].values.astype(int)[idx], device=data.edge_index.device
                )
            else:
                data[prefix + attr] = torch.tensor(
                    df[attr].values.astype(float)[idx], device=data.edge_index.device
                )
        else:
            # if the attribute is not iterable, convert it to a string
            data[prefix + attr] = np.array(df[attr].values.astype(str)[idx])
    else:
        data[prefix + attr] = torch.tensor(df[attr].values[idx], device=data.edge_index.device)


def df_to_graph(df: pd.DataFrame, is_undirected: bool = False, multiedges: bool = False, num_nodes: int | None = None) -> Graph:
    """Reads a network from a pandas data frame.

    The data frame is expected to have a minimum of two columns
    that give the source and target nodes of edges. Additional columns in the
    data frame will be mapped to edge attributes.

    Args:

        df: pandas.DataFrame

            A data frame with rows containing edges and optional edge attributes. If the
            data frame contains column names, the source and target columns must be called
            'v' and 'w' respectively. If no column names are used the first two columns
            are interpreted as source and target.

        is_undirected: Optional[bool]=True

            whether or not to interpret edges as undirected

        multiedges: Optional[bool]=False

            whether or not to allow multiple edges between the same node pair. By
            default multi edges are ignored.

    Example:
        ```py

        import pathpyG as pp
        import pandas as pd

        df = pd.DataFrame({
            'v': ['a', 'b', 'c'],
            'w': ['b', 'c', 'a'],
            'edge_weight': [1.0, 5.0, 2.0]
            })
        g = pp.io.df_to_graph(df)
        print(n)
        ```
    """
    # assign column names if no header is present
    no_header = all(isinstance(x, int) for x in df.columns.values.tolist())

    if no_header:
        # interpret first two columns as source and target
        col_names = ["v", "w"]
        # interpret remaining columns as edge attributes
        for i in range(2, len(df.columns.values.tolist())):
            col_names += ["edge_attr_{0}".format(i - 2)]
        df.columns = col_names

    edge_df = df[["v", "w"]].drop_duplicates()
    if not multiedges and (len(edge_df) != len(df)):
        print("Data frame contains multiple edges, but multiedges is set to False. Removing duplicates.")
        df = df.drop_duplicates(subset=["v", "w"])

    mapping = IndexMap(node_ids=np.unique(df[["v", "w"]].values))
    data = Data(
        edge_index=mapping.to_idxs(df[["v", "w"]].values.T),
        num_nodes=num_nodes if num_nodes is not None else mapping.node_ids.shape[0],
    )
    cols = df.columns.tolist()
    cols.remove("v")
    cols.remove("w")
    for col in cols:
        if col.startswith("edge_"):
            prefix = ""
        else:
            prefix = "edge_"

        _parse_df_column(
            df=df,
            data=data,
            attr=col,
            prefix=prefix
        )
    g = Graph(data=data, mapping=mapping)
    if is_undirected:
        g = g.to_undirected()
    return g


def add_node_attributes(df: pd.DataFrame, g: Graph):
    """Add node attributes from pandas data frame to existing `Graph`.

    Add node attributes from pandas data frame to existing graph, where node
    IDs or indices are given in column `v` and node attributes x are given in columns `node_x`.

    Args:
        df: A DataFrame with rows containing nodes and optional node attributes.
        g: The graph to which the node attributes should be added.
    """
    if "v" in df:
        print("Mapping node attributes based on node names in column `v`")
        attributed_nodes = list(df["v"])
    elif "index" in df:
        print("Mapping node attributes based on node indices in column `index`")
        attributed_nodes = list(df["index"])
    else:
        print("Data frame must either have `index` or `v` column")
        return

    # check for duplicated node attributes
    if len(set(attributed_nodes)) < len(attributed_nodes):
        print("data frame cannot contain multiple attribute values for single node")
        return

    # check for difference between nodes in graph and nodes in attributes
    if "v" in df:
        if set(attributed_nodes) != set([v for v in g.nodes]):
            print("Mismatch between nodes in DataFrame and nodes in graph")
            return

        # get indices of nodes in tensor
        node_idx = g.mapping.to_idxs(attributed_nodes)
    else:
        if set(attributed_nodes) != set([i for i in range(g.n)]):
            print("Mismatch between nodes in DataFrame and nodes in graph")
            return

        # get indices of nodes in tensor
        node_idx = attributed_nodes

    # assign node property tensors
    for attr in df.columns:

        # skip node column
        if attr == "v" or attr == "index":
            continue

        # prefix attribute names that are not already prefixed
        if attr.startswith("node_"):
            prefix = ""
        else:
            prefix = "node_"

        _parse_df_column(
            df=df,
            data=g.data,
            idx=node_idx,
            attr=attr,
            prefix=prefix,
        )


def add_edge_attributes(df: pd.DataFrame, g: Graph, time_attr: str | None = None) -> None:
    """Add (temporal) edge attributes from pandas data frame to existing `Graph`.

    Add edge attributes from `pandas.DataFrame` to existing `Graph`, where source/target node
    IDs are given in columns `v` and `w`  and edge attributes x are given in columns `edge_x`.
    If `time_attr` is not None, the dataframe is expected to contain temporal data with a timestamp
    in a column named as specified in `time_attr`.

    Args:
        df: A DataFrame with rows containing edges and optional edge attributes.
        g: The graph to which the edge attributes should be added.
        time_attr: If not None, the name of the column containing time stamps for temporal edges.
    """
    assert "v" in df and "w" in df, "Data frame must have columns `v` and `w` for source and target nodes"

    # extract indices of source/target node of edges
    src = g.mapping.to_idxs(df["v"].tolist())
    tgt = g.mapping.to_idxs(df["w"].tolist())

    edge_attrs = list(df.columns)
    edge_attrs.remove("v")
    edge_attrs.remove("w")

    if time_attr is not None:
        assert time_attr in df, f"Data frame must have column `{time_attr}` for time stamps"

        time = df[time_attr].values
        edge_attrs.remove(time_attr)

        # find indices of edges in edge_index
        edge_idx = []
        for src_i, tgt_i, time_i in zip(src, tgt, time):
            matching_idx = torch.where(
                (g.data.edge_index[0, :] == src_i) & (g.data.edge_index[1, :] == tgt_i) & (g.data.time == time_i)
            )[0]
            if matching_idx.numel() == 1:
                edge_idx.append(matching_idx.item())
            else:
                # if the edge is not unique, raise a warning
                if matching_idx.numel() > 1:
                    # if there are multiple edges, take the first one
                    edge_idx.append(matching_idx[0].item()) 
                warnings.warn(f"Edge ({src_i}, {tgt_i}) exists {matching_idx.numel()} times in the graph", stacklevel=2)
    else:
        # find indices of edges in edge_index
        edge_idx = []
        for src_i, tgt_i in zip(src, tgt):
            matching_idx = torch.where((g.data.edge_index[0, :] == src_i) & (g.data.edge_index[1, :] == tgt_i))[0]
            assert (
                matching_idx.numel() == 1
            ), f"Edge ({src_i}, {tgt_i}) either does not exist or is duplicated in the graph"
            edge_idx.append(matching_idx.item())

    for attr in edge_attrs:
        if attr.startswith("edge_"):
            prefix = ""
        else:
            prefix = "edge_"

        # parse column and add to graph
        _parse_df_column(
            df=df,
            data=g.data,
            idx=edge_idx,
            attr=attr,
            prefix=prefix,
        )


def df_to_temporal_graph(
    df: pd.DataFrame, is_undirected: bool = False, multiedges: bool = False, timestamp_format="%Y-%m-%d %H:%M:%S", time_rescale=1, num_nodes: int | None = None
) -> TemporalGraph:
    """Reads a temporal graph from a pandas data frame.

    The data frame is expected to have a minimum of two columns `v` and `w`
    that give the source and target nodes of edges. Additional column names to
    be used can be configured in `config.cfg` as `v_synonyms` and `w`
    synonyms. The time information on edges can either be stored in an
    additional `timestamp` column (for instantaneous interactions) or in two
    columns `start`, `end` or `timestamp`, `duration` respectively for networks
    where edges appear and exist for a certain time. Synonyms for those column
    names can be configured in config.cfg.  Each row in the data frame is
    mapped to one temporal edge. Additional columns in the data frame will be
    mapped to edge attributes.

    Args:
        df: pandas.DataFrame with rows containing time-stamped edges and optional edge
        attributes.
        timestamp_format: timestamp format
        time_rescale: time stamp rescaling factor
        **kwargs: Arbitrary keyword arguments that will be set as network-level attributes.

    Example:
    ```py

    import pathpyG as pp
    import pandas as pd
    df = pd.DataFrame({
        'v': ['a', 'b', 'c'],
        'w': ['b', 'c', 'a'],
        't': [1, 2, 3]})
    g = pp.io.df_to_temporal_graph(df)
    print(g)

    df = pd.DataFrame([
        ['a', 'b', 'c'],
        ['b', 'c', 'a'],
        [1, 2, 3]
        ])
    g = pp.io.df_to_temporal_graph(df)
    print(g)
    ```
    """
    # assign column names if no header is present
    no_header = all(isinstance(x, int) for x in df.columns.values.tolist())

    if no_header:
        # interpret first two columns as source and target
        col_names = ["v", "w", "t"]
        # interpret remaining columns as edge attributes
        for i in range(3, len(df.columns.values.tolist())):
            col_names += ["edge_attr_{0}".format(i - 2)]
        df.columns = col_names

    # optionally parse time stamps
    if df["t"].dtype == "object":
        # convert time stamps to seconds since epoch
        df["t"] = pd.to_datetime(df["t"], format=timestamp_format)
        # rescale time stamps
        df["t"] = df["t"].astype("int64") // time_rescale
    elif df["t"].dtype == "int64" or df["t"].dtype == "float64":
        # rescale time stamps
        df["t"] = df["t"] // time_rescale
    elif pd.api.types.is_datetime64_any_dtype(df["t"]):
        df["t"] = df["t"].astype("int64") // time_rescale
    else:
        raise ValueError(
            "Column `t` must be of type `object`, `int64`, `float64`, or a datetime type. "
            f"Found {df['t'].dtype} instead."
        )

    if not multiedges:
        df = df.drop_duplicates(subset=["v", "w", "t"])

    mapping = IndexMap(node_ids=np.unique(df[["v", "w"]].values))
    data = Data(
        edge_index=mapping.to_idxs(df[["v", "w"]].values.T),
        time=torch.tensor(df["t"].values),
        num_nodes=num_nodes if num_nodes is not None else mapping.node_ids.shape[0],
    )
    cols = df.columns.tolist()
    cols.remove("v")
    cols.remove("w")
    for col in cols:
        if col.startswith("edge_"):
            prefix = ""
        else:
            prefix = "edge_"

        _parse_df_column(
            df=df,
            data=data,
            attr=col,
            prefix=prefix
        )
    g = TemporalGraph(data=data, mapping=mapping)
    
    if is_undirected:
        g = g.to_undirected()
    
    return g


def graph_to_df(graph: Graph, node_indices: Optional[bool] = False) -> pd.DataFrame:
    """Returns a pandas data frame for a given graph.

    Returns a pandas dataframe data that contains all edges including edge
    attributes. Node and network-level attributes are not included. To
    facilitate the import into network analysis tools that only support integer
    node identifiers, node uids can be replaced by a consecutive, zero-based
    index.

    Args:
        graph: The graph to export as pandas DataFrame
        node_indices: whether nodes should be exported as integer indices

    Example:
    ```py
    import pathpyG as pp

    n = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('c', 'a')])
    df = pp.io.to_dataframe(n)
    print(df)
    ```
    """
    df = pd.DataFrame()

    for v, w in graph.edges:
        if node_indices:
            v = graph.mapping.to_idx(v)
            w = graph.mapping.to_idx(w)
        edge_frame = pd.DataFrame.from_dict({"v": [v], "w": [w]})
        df = pd.concat([df, edge_frame], ignore_index=True, sort=False)

    edge_attribute_df = pd.DataFrame.from_dict({a: graph.data[a] for a in graph.edge_attrs()})
    df = pd.concat([df, edge_attribute_df], axis=1)
    return df


def temporal_graph_to_df(graph: TemporalGraph, node_indices: Optional[bool] = False) -> pd.DataFrame:
    """Returns a pandas data frame for a given temporal graph.

    Returns a pandas dataframe data that contains all edges including edge
    attributes. Node and network-level attributes are not included. To
    facilitate the import into network analysis tools that only support integer
    node identifiers, node uids can be replaced by a consecutive, zero-based
    index.

    Args:
        graph: The graph to export as pandas DataFrame
        node_indices: whether nodes should be exported as integer indices

    Example:
    ```py
    import pathpyG as pp

    n = pp.TemporalGraph.from_edge_list([('a', 'b', 1), ('b', 'c', 2), ('c', 'a', 3)])
    df = pp.io.to_df(n)
    print(df)
    ```
    """
    df = pd.DataFrame()

    # export temporal graph
    for v, w, t in graph.temporal_edges:
        if node_indices:
            v = graph.mapping.to_idx(v)
            w = graph.mapping.to_idx(w)
        edge_frame = pd.DataFrame.from_dict({"v": [v], "w": [w], "t": [t]})
        # data = pd.DataFrame.from_dict(
        #    {k: [v] for k, v in edge.attributes.items()})
        # edge_frame = pd.concat([edge_frame, data], axis=1)
        df = pd.concat([edge_frame, df], ignore_index=True, sort=False)
    return df


def read_csv_graph(
    filename: str,
    sep: str = ",",
    header: bool = True,
    is_undirected: bool = False,
    multiedges: bool = False,
    **kwargs: Any,
) -> Graph:
    """Reads a Graph or TemporalGraph from a csv file. To read a temporal graph, the csv file must have
    a header with column `t` containing time stamps of edges

    Args:
        loops:  whether or not to add self_loops
        directed: whether or not to intepret edges as directed
        multiedges: whether or not to add multiple edges
        sep: character separating columns in the csv file
        header: whether or not the first line of the csv file is interpreted as header with column names
        timestamp_format: format of timestamps
        time_rescale: rescaling of timestamps

    Example:
        ```py
        import pathpyG as pp

        g = pp.io.read_csv('example_graph.csv')
        g = pp.io.read_csv('example_temporal_graph.csv')
        ```
    """
    if header:
        df = pd.read_csv(filename, header=0, sep=sep)
    else:
        df = pd.read_csv(filename, header=None, sep=sep)

    return df_to_graph(df, is_undirected=is_undirected, multiedges=multiedges, **kwargs)


def read_csv_temporal_graph(
    filename: str,
    sep: str = ",",
    header: bool = True,
    is_undirected: bool = True,
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    time_rescale: int = 1,
    **kwargs: Any,
) -> TemporalGraph:
    """Reads a TemporalGraph from a csv file that minimally has three columns
    containin source, target and time.

    Args:
        sep: character separating columns in the csv file
        header: whether or not the first line of the csv file is interpreted as header with column names
        directed: whether or not to intepret edges as directed
        timestamp_format: format of timestamps
        time_rescale: rescaling of timestamps

    Example:
        ```py
        import pathpyG as pp

        g = pp.io.read_csv('example_temporal_graph.csv')
        ```
    """
    if header:
        df = pd.read_csv(filename, header=0, sep=sep)
    else:
        df = pd.read_csv(filename, header=None, sep=sep)
    return df_to_temporal_graph(
        df, is_undirected=is_undirected, timestamp_format=timestamp_format, time_rescale=time_rescale, **kwargs
    )


def write_csv(
    graph: Union[Graph, TemporalGraph], path_or_buf: Any = None, node_indices: bool = False, **pdargs: Any
) -> None:
    """Stores all edges including edge attributes in a csv file."""
    if isinstance(graph, TemporalGraph):
        frame = temporal_graph_to_df(graph=graph, node_indices=node_indices)
    else:
        frame = graph_to_df(graph=graph, node_indices=node_indices)
    frame.to_csv(path_or_buf=path_or_buf, index=False, **pdargs)
