from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from collections import Counter

import pandas as pd
import torch
import numpy as np

import datetime
from time import mktime

from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.utils.config import config


def _check_column_name(frame: pd.DataFrame, name: str, synonyms: list) -> pd.DataFrame:
    """Helper function to check column names and change them if needed."""
    if name not in frame.columns:
        for col in frame.columns:
            if col in synonyms:
                frame.rename(columns={col: name}, inplace=True)
                continue
    return frame


def df_to_graph(df: pd.DataFrame, is_undirected: bool = False, multiedges: bool = False, **kwargs: Any) -> Graph:
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

    df["v"] = df["v"].astype(str)
    df["w"] = df["w"].astype(str)

    edges: list = []
    edge_set: set = set()

    # counter for multiple edges
    counter: Counter = Counter()

    for row in df.to_dict(orient="records"):
        _v, _w = row.pop("v"), row.pop("w")

        # check if edge was already generated
        if (_v, _w) in edge_set and not multiedges:
            counter[(_v, _w)] += 1
        else:
            # add edge
            edges.append((_v, _w))
            edge_set.add((_v, _w))

    # check for multi-edges
    if len(counter) > 0:
        print(
            "%i edges existed already "
            "and were not considered. "
            "To capture those edges, consider creating "
            "a multiedge and/or directed network.",
            sum(counter.values()),
        )

    # create graph
    g = Graph.from_edge_list(edges, is_undirected=is_undirected, **kwargs)

    # assign edge attributes
    add_edge_attributes(df, g)
    return g


def add_node_attributes(df: pd.DataFrame, g: Graph):
    """Add node attributes from pandas data frame to existing graph, where node
    IDs or indices are given in column `v` and node attributes x are given in columns `node_x`
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
        node_idx = [g.mapping.to_idx(x) for x in df["v"]]
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
        prefix = ""
        if not attr.startswith("node_"):
            prefix = "node_"

        # eval values for array-valued attributes
        try:
            values = np.array([eval(x) for x in df[attr].values])
            g.data[prefix + attr] = torch.from_numpy(values[node_idx]).to(device=g.data.edge_index.device)
            continue
        except:
            pass

        # try to directly construct tensor for scalar values
        try:
            g.data[prefix + attr] = torch.from_numpy(df[attr].values[node_idx]).to(device=g.data.edge_index.device)
            continue
        except:
            pass

        # numpy array of strings
        try:
            g.data[prefix + attr] = np.array(df[attr].values.astype(str)[node_idx])
        except:
            t = df[attr].dtype
            print(f"Could not assign node attribute {attr} of type {t}")


def add_edge_attributes(df: pd.DataFrame, g: Graph) -> None:
    """Add edge attributes from pandas data frame to existing graph, where source/target node
    IDs are given in columns `v` and `w`  and edge attributes x are given in columns `edge_x`
    """
    if "t" in df:
        if "v" not in df or "w" not in df or "t" not in df:
            print("data frame must have columns `v` and `w` and `t`")
            return

        attributed_edges = list(zip(df["v"], df["w"], df["t"]))


        # extract indices of source/target node of edges
        src = [g.mapping.to_idx(x) for x in df["v"]]
        tgt = [g.mapping.to_idx(x) for x in df["w"]]
        time = [x for x in df["t"]]

        # unique index for each edge independent of v,w,t because there exist temporal networks with duplicated temporal edges
        edge_idx = list(range(len(src))) 

        #sort the edge_index for the case that the data_frame is not sorted
        paired = list(zip(time, edge_idx))
        paired.sort(key=lambda x: x[0])
        edge_idx = [idx for _, idx in paired]

        for attr in df.columns:
            if attr != "v" and attr != "w" and attr != "t":
                prefix = ""
                if not attr.startswith("edge_"):
                    prefix = "edge_"
                
                # eval values for array-valued attributes
                try:
                    values = np.array([eval(x) for x in df[attr].values])
                    
                    g.data[prefix + attr] = torch.from_numpy(values[edge_idx]).to(device=g.data.edge_index.device)
                    continue
                except:
                    pass
                
                # try to directly construct tensor for scalar values
                try:
                    g.data[prefix + attr] = torch.from_numpy(df[attr].values[edge_idx]).to(device=g.data.edge_index.device)
                    continue
                except:
                    pass

                # numpy array of strings
                try:
                    g.data[prefix + attr] = np.array(df[attr].values.astype(str)[edge_idx])
                except:
                    t = df[attr].dtype
                    print(f"Could not assign edge attribute {attr} of type {t}")





    else:
        if "v" not in df or "w" not in df:
            print("data frame must have columns `v` and `w`")
            return

        attributed_edges = list(zip(df["v"], df["w"]))

        # check for duplicated edge attributes
        if len(set(attributed_edges)) < len(attributed_edges):
            print("data frame contains multiple attribute values for single edge")
            return

        # check for difference between edges in graph and edges in attributes
        if set(attributed_edges) != set([(v, w) for v, w in g.edges]):
            print("Mismatch between edges in DataFrame and edges in graph")
            return

        # extract indices of source/target node of edges
        src = [g.mapping.to_idx(x) for x in df["v"]]
        tgt = [g.mapping.to_idx(x) for x in df["w"]]

        # find indices of edges in edge_index
        edge_idx = []
        for i in range(len(src)):
            x = torch.where((g.data.edge_index[0, :] == src[i]) & (g.data.edge_index[1, :] == tgt[i]))[0].item()
            edge_idx.append(x)
        for attr in df.columns:
            if attr != "v" and attr != "w":
                prefix = ""
                if not attr.startswith("edge_"):
                    prefix = "edge_"

                # eval values for array-valued attributes
                try:
                    values = np.array([eval(x) for x in df[attr].values])
                    g.data[prefix + attr] = torch.from_numpy(values[edge_idx]).to(device=g.data.edge_index.device)
                    continue
                except:
                    pass

                # try to directly construct tensor for scalar values
                try:
                    g.data[prefix + attr] = torch.from_numpy(df[attr].values[edge_idx]).to(device=g.data.edge_index.device)
                    continue
                except:
                    pass

                # numpy array of strings
                try:
                    g.data[prefix + attr] = np.array(df[attr].values.astype(str)[edge_idx])
                except:
                    t = df[attr].dtype
                    print(f"Could not assign edge attribute {attr} of type {t}")

                # g.data[prefix+attr] = df[attr].values[edge_idx]


def df_to_temporal_graph(
    df: pd.DataFrame, is_undirected: bool = False, timestamp_format="%Y-%m-%d %H:%M:%S", time_rescale=1, **kwargs: Any
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

    tedges = []
    for row in df.to_dict(orient="records"):
        _v, _w, _t = str(row.pop("v")), str(row.pop("w")), str(row.pop("t"))
        try:
            t = float(_t)
        except:
            # if time stamp is a string, use timestamp_format to convert
            # it to UNIX timestamp
            x = datetime.datetime.strptime(_t, timestamp_format)
            t = int(mktime(x.timetuple()))
        tedges.append((_v, _w, int(t / time_rescale)))

    g = TemporalGraph.from_edge_list(tedges, **kwargs)
    add_edge_attributes(df, g)
    if is_undirected:
        return g.to_undirected()
    else:
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
