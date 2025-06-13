from __future__ import annotations
from typing import Any, Optional, Union

import ast
import re
import logging

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data

from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.core.index_map import IndexMap
from pathpyG.utils.convert import to_numpy

logger = logging.getLogger("root")

# Regex to check if the attribute is iterable (e.g., list or tuple), a number (int or float), or an integer.
_iterable_re = re.compile(r"^\s*[\[\(].*[\]\)]\s*$")
_number_re = re.compile(r"^\s*[+-]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+-]?\d+)?\s*$")
_integer_re = re.compile(r"^\s*[+-]?\d+\s*$")


def _parse_timestamp(df: pd.DataFrame, timestamp_format: str = "%Y-%m-%d %H:%M:%S", time_rescale: int = 1) -> None:
    """Parse time stamps in a DataFrame.

    Parses the time stamps in the DataFrame and rescales using the given time rescale factor.
    The time stamps are expected to be in a column named `t`. If the column is of type `object`, it is assumed to
    contain time stamps in the specified format.

    Args:
        df: The DataFrame containing the time stamps in a column named `t`.
        timestamp_format: The format of the time stamps in the `t` column.
        time_rescale: The factor by which to rescale the time stamps. Defaults to 1, meaning no rescaling.
    """
    # optionally parse time stamps
    if df["t"].dtype == "object" and isinstance(df["t"].values[0], str):
        # convert time stamps to seconds since epoch
        df["t"] = pd.to_datetime(df["t"], format=timestamp_format)
        # rescale time stamps
        df["t"] = df["t"].astype("int64") // time_rescale
        df["t"] = df["t"] - df["t"].min()  # rescale to start at 0
    elif df["t"].dtype == "int64" or df["t"].dtype == "float64":
        # rescale time stamps
        df["t"] = df["t"] // time_rescale
    elif pd.api.types.is_datetime64_any_dtype(df["t"]):
        df["t"] = df["t"].astype("int64") // time_rescale
        df["t"] = df["t"] - df["t"].min()  # rescale to start at 0
    else:
        raise ValueError(
            "Column `t` must be of type `object`, `int64`, `float64`, or a datetime type. "
            f"Found {df['t'].dtype} instead."
        )


def _parse_df_column(
    df: pd.DataFrame, data: Data, attr: str, idx: list | np.ndarray | None = None, prefix: str = ""
) -> None:
    """Parse a column in a DataFrame and add it as an attribute to the graph.

    Parses a column in a DataFrame and adds it as an attribute to the graph's data object. We assume that the attribute
    in the DataFrame is ordered in the same way as the nodes/edges in the graph if `idx` is not provided. If `idx` is
    provided, the order of the attribute values is determined by the indices in `idx`.

    Args:
        df: The DataFrame containing the attribute. Attributes are expected to be numeric, string, or iterable types.
        data: The Data object of the graph to which the attribute should be added.
        attr: The name of the attribute column in the DataFrame.
        idx: Indices specifying the order of the attribute values. If None, all values are used in the given order.
        prefix: A prefix to be added to the attribute name in the Data object, e.g., "edge_" or "node_".
    """
    # if idx is None, use all indices in the given order
    if idx is None:
        idx = np.arange(len(df))

    # check if the attribute is a string, list, tuple, etc.
    if df[attr].dtype == "object":
        if isinstance(df[attr].values[0], str):
            # if the attribute is a string, check if it is iterable or numeric
            if _iterable_re.match(str(df[attr].values[0])):
                # if the attribute is a string that can be converted to an iterable, convert it to a tensor
                data[prefix + attr] = torch.tensor(
                    [ast.literal_eval(x) for x in df[attr].values[idx]], device=data.edge_index.device
                )
            elif _number_re.match(str(df[attr].values[0])):
                # if the attribute is a number, convert it to a tensor
                if _integer_re.match(str(df[attr].values[0])):
                    data[prefix + attr] = torch.tensor(df[attr].values.astype(int)[idx], device=data.edge_index.device)
                else:
                    data[prefix + attr] = torch.tensor(
                        df[attr].values.astype(float)[idx], device=data.edge_index.device
                    )
            else:
                # if the attribute is not iterable, convert it to a string
                data[prefix + attr] = np.array(df[attr].values.astype(str)[idx])
        elif isinstance(df[attr].values[0], (list, tuple)):
            data[prefix + attr] = torch.tensor([np.array(x) for x in df[attr].values[idx]])
        else:
            raise ValueError(f"Unsupported data type for attribute '{attr}': {type(df[attr].values[0])}")
    else:
        # if the attribute is numeric, convert it to a tensor directly
        data[prefix + attr] = torch.tensor(df[attr].values[idx], device=data.edge_index.device)


def df_to_graph(
    df: pd.DataFrame, is_undirected: bool = False, multiedges: bool = False, num_nodes: int | None = None
) -> Graph:
    """Reads a network from a pandas data frame.

    The data frame is expected to have a minimum of two columns
    that give the source and target nodes of edges. Additional columns in the
    data frame will be mapped to edge attributes.

    Args:
        df: A data frame with rows containing edges and optional edge attributes. If the
            data frame contains column names, the source and target columns must be called
            'v' and 'w' respectively. If no column names are used the first two columns
            are interpreted as source and target.
        is_undirected: Whether or not to interpret edges as undirected.
        multiedges: Whether or not to allow multiple edges between the same node pair. By
            default multi edges are ignored.
        num_nodes: The number of nodes in the graph. If None, the number of unique nodes
            in the data frame is used.

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
            col_names += [f"edge_attr_{i - 2}"]
        df.columns = col_names

    # optionally remove multiedges
    if not multiedges and df[["v", "w"]].duplicated().any():
        logger.debug("Data frame contains multiple edges, but multiedges is set to False. Removing duplicates.")
        df = df.drop_duplicates(subset=["v", "w"])

    # Create index mapping and data object
    mapping = IndexMap(node_ids=np.unique(df[["v", "w"]].values).tolist())
    data = Data(
        edge_index=mapping.to_idxs(df[["v", "w"]].values.T),
        num_nodes=num_nodes if num_nodes is not None else mapping.node_ids.shape[0],  # type: ignore
    )

    # Parse all columns except 'v' and 'w' as edge attributes
    cols = df.columns.tolist()
    cols.remove("v")
    cols.remove("w")
    for col in cols:
        if col.startswith("edge_"):
            prefix = ""
        else:
            prefix = "edge_"

        _parse_df_column(df=df, data=data, attr=col, prefix=prefix)

    # Create graph object
    g = Graph(data=data, mapping=mapping)
    # If the graph should be undirected, convert it to an undirected graph
    if is_undirected:
        g = g.to_undirected()

    return g


def add_node_attributes(df: pd.DataFrame, g: Graph):
    """Add node attributes from `DataFrame` to existing `Graph`.

    Add node attributes from `pandas.DataFrame` to existing graph, where node
    IDs or indices are given in column `v` and node attributes x are given in columns `node_x`.

    Args:
        df: A DataFrame with rows containing nodes and optional node attributes.
        g: The graph to which the node attributes should be added.
    """
    if "v" in df:
        logger.debug("Mapping node attributes based on node names in column `v`")
        attributed_nodes = list(df["v"])
    elif "index" in df:
        logger.debug("Mapping node attributes based on node indices in column `index`")
        attributed_nodes = list(df["index"])
    else:
        raise ValueError("DataFrame must either have `index` or `v` column")

    # check for duplicated node attributes
    if len(set(attributed_nodes)) < len(attributed_nodes):
        raise ValueError("DataFrame cannot contain multiple attribute values for single node")

    # check for difference between nodes in graph and nodes in attributes
    if "v" in df:
        if set(attributed_nodes) != set([v for v in g.nodes]):
            raise ValueError("Mismatch between nodes in DataFrame and nodes in graph")

        # get indices of nodes in tensor
        node_idx = g.mapping.to_idxs(attributed_nodes).tolist()
    else:
        if set(attributed_nodes) != set([i for i in range(g.n)]):
            raise ValueError("Mismatch between nodes in DataFrame and nodes in graph")

        # get indices of nodes in tensor
        node_idx = attributed_nodes

    # assign node property tensors
    cols = [attr for attr in df.columns if attr not in ["v", "index"]]
    for attr in cols:
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

    # check for non-existent nodes
    node_ids = set(df["v"]).union(set(df["w"]))
    if not node_ids.issubset(set(g.nodes)):
        raise ValueError(
            f"DataFrame contains nodes {node_ids - set(g.nodes)} that do not exist in the graph. "
            "Please ensure all nodes in the DataFrame are present in the graph."
        )

    # check if the number of edges in the data frame is consistent with the graph
    if g.m != len(df):
        raise ValueError(
            f"DataFrame contains {len(df)} edges, but the graph has {g.m} edges. "
            "Please ensure the DataFrame matches the number of edges in the graph."
        )

    # extract indices of source/target node of edges
    src = g.mapping.to_idxs(df["v"].tolist())
    tgt = g.mapping.to_idxs(df["w"].tolist())

    edge_attrs = [attr for attr in df.columns if attr not in ["v", "w"]]

    if time_attr is not None:
        assert time_attr in df, f"Data frame must have column `{time_attr}` for time stamps"

        time = df[time_attr].values
        edge_attrs.remove(time_attr)

        # find indices of edges in temporal edge_index
        edge_idx = []
        for src_i, tgt_i, time_i in zip(src, tgt, time):
            edge = g.tedge_to_index.get((src_i.item(), tgt_i.item(), time_i.item()), None)  # type: ignore
            if edge is None:
                raise ValueError(
                    f"Edge ({src_i.item()}, {tgt_i.item()}) does not exist at time {time_i.item()} in the graph."
                )
            edge_idx.append(edge)
    else:
        # find indices of edges in edge_index
        edge_idx = []
        for src_i, tgt_i in zip(src, tgt):
            edge = g.edge_to_index.get((src_i.item(), tgt_i.item()), None)
            if edge is None:
                raise ValueError(f"Edge ({src_i.item()}, {tgt_i.item()}) does not exist in the graph.")
            edge_idx.append(edge)

    for attr in edge_attrs:
        if attr.startswith("edge_"):
            prefix = ""
        else:
            prefix = "edge_"

        # parse column and add to graph
        _parse_df_column(
            df=df.iloc[edge_idx],
            data=g.data,
            attr=attr,
            prefix=prefix,
        )


def df_to_temporal_graph(
    df: pd.DataFrame,
    multiedges: bool = False,
    timestamp_format="%Y-%m-%d %H:%M:%S",
    time_rescale=1,
    num_nodes: int | None = None,
) -> TemporalGraph:
    """Read a temporal graph from a DataFrame.

    The DataFrame is expected to have a minimum of two columns `v` and `w`
    that give the source and target nodes of edges. Each row in the DataFrame is
    mapped to one temporal edge. Additional columns in the DataFrame will be
    mapped to edge attributes.

    Args:
        df: pandas.DataFrame with rows containing time-stamped edges and optional edge
            attributes.
        multiedges: Whether or not to allow multiple edges between the same node pair. By
            default multi edges are ignored.
        timestamp_format: The format of the time stamps in the `t` column.
        time_rescale: The factor by which to rescale the time stamps. Defaults to 1, meaning no rescaling.
        num_nodes: The number of nodes in the graph. If None, the number of unique nodes
            in the DataFrame is used.

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

    # parse the time stamp column "t"
    _parse_timestamp(df=df, timestamp_format=timestamp_format, time_rescale=time_rescale)

    # optionally remove multiedges
    if not multiedges:
        df = df.drop_duplicates(subset=["v", "w", "t"])

    # Create index mapping and data object
    mapping = IndexMap(node_ids=np.unique(df[["v", "w"]].values))
    data = Data(
        edge_index=mapping.to_idxs(df[["v", "w"]].values.T),
        time=torch.tensor(df["t"].values),
        num_nodes=num_nodes if num_nodes is not None else mapping.node_ids.shape[0],  # type: ignore
    )

    # add edge attributes
    cols = [col for col in df.columns if col not in ["v", "w", "t"]]
    for col in cols:
        if col.startswith("edge_"):
            prefix = ""
        else:
            prefix = "edge_"

        _parse_df_column(df=df, data=data, attr=col, prefix=prefix)

    # Create temporal graph object
    g = TemporalGraph(data=data, mapping=mapping)

    return g


def graph_to_df(graph: Graph, node_indices: Optional[bool] = False) -> pd.DataFrame:
    """Return a DataFrame for a given graph.

    Returns a `pandas.DataFrame` that contains all edges including edge
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
    if node_indices:
        vs = to_numpy(graph.data.edge_index[0])
        ws = to_numpy(graph.data.edge_index[1])
    else:
        vs = graph.mapping.to_ids(to_numpy(graph.data.edge_index[0]))
        ws = graph.mapping.to_ids(to_numpy(graph.data.edge_index[1]))
    df = pd.DataFrame({**{"v": vs, "w": ws}, **{a: graph.data[a].tolist() for a in graph.edge_attrs()}})

    return df


def temporal_graph_to_df(graph: TemporalGraph, node_indices: Optional[bool] = False) -> pd.DataFrame:
    """Return a DataFrame for a given temporal graph.

    Returns a `pandas.DataFrame` that contains all edges including edge
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
    if node_indices:
        vs = to_numpy(graph.data.edge_index[0])
        ws = to_numpy(graph.data.edge_index[1])
    else:
        vs = graph.mapping.to_ids(to_numpy(graph.data.edge_index[0]))
        ws = graph.mapping.to_ids(to_numpy(graph.data.edge_index[1]))
    df = pd.DataFrame(
        {
            **{"v": vs, "w": ws, "t": graph.data.time.tolist()},
            **{a: graph.data[a].tolist() for a in graph.edge_attrs()},
        }
    )

    return df


def read_csv_graph(
    filename: str,
    sep: str = ",",
    header: bool = True,
    is_undirected: bool = False,
    multiedges: bool = False,
    **kwargs: Any,
) -> Graph:
    """Read a `Graph` from a csv file.

    This method reads a graph from a `.csv`-file and converts it to a
    `Graph` object. To read a temporal graph, the csv file must have
    a header with column `t` containing time stamps of edges

    Args:
        filename: The path to the csv file containing the graph data.
        sep: character separating columns in the csv file
        header: whether or not the first line of the csv file is interpreted as header with column names
        is_undirected: whether or not to interpret edges as undirected
        multiedges: whether or not to allow multiple edges between the same node pair. By default multi edges are
            ignored.
        **kwargs: Additional keyword arguments passed to the `df_to_graph` function.

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
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    time_rescale: int = 1,
    **kwargs: Any,
) -> TemporalGraph:
    """Read a `TemporalGraph` from a csv file.

    This method reads a temporal graph from a `.csv`-file and converts it to a
    `TemporalGraph` object. The csv file is expected to have a header with columns
    `v`, `w`, and `t` containing source nodes, target nodes, and time stamps of edges,
    respectively. Additional columns in the csv file will be interpreted as edge attributes.

    Args:
        filename: The path to the csv file containing the temporal graph data.
        sep: character separating columns in the csv file
        header: whether or not the first line of the csv file is interpreted as header with column names
        timestamp_format: The format of the time stamps in the `t` column.
        time_rescale: The factor by which to rescale the time stamps. Defaults to 1, meaning no rescaling.
        **kwargs: Additional keyword arguments passed to the `df_to_temporal_graph` function.

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
    return df_to_temporal_graph(df, timestamp_format=timestamp_format, time_rescale=time_rescale, **kwargs)


def write_csv(graph: Union[Graph, TemporalGraph], node_indices: bool = False, **pdargs: Any) -> None:
    """Store all edges including edge attributes in a csv file.

    This method stores a `Graph` or `TemporalGraph` as a `.csv` file. The csv file
    will contain all edges including edge attributes. Node and network-level attributes
    are not included. To facilitate the import into network analysis tools that only
    support integer node identifiers, node uids can be replaced by a consecutive,
    zero-based index.

    Args:
        graph: The graph to export as pandas DataFrame
        node_indices: whether nodes should be exported as integer indices
        **pdargs: Additional keyword arguments passed to `pandas.DataFrame.to_csv`.
    """
    if isinstance(graph, TemporalGraph):
        frame = temporal_graph_to_df(graph=graph, node_indices=node_indices)
    else:
        frame = graph_to_df(graph=graph, node_indices=node_indices)
    frame.to_csv(index=False, **pdargs)
