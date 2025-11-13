"""Module for database I/O operations."""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.io.pandas import add_node_attributes, df_to_graph, df_to_temporal_graph, graph_to_df, temporal_graph_to_df

logger = logging.getLogger("root")


def read_sql(
    db_path: str,
    edge_table: str = "edges",
    node_table: str | None = None,
    source_name: str = "source",
    target_name: str = "target",
    time_name: str | None = None,
    node_name: str = "node_id",
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    time_rescale: int = 1,
) -> Graph:
    """Read a graph from an SQL database file.

    The function reads edge and node data from specified tables in the database
    and constructs a [graph][pathpyG.Graph] or a [temporal graph][pathpyG.TemporalGraph]
    if a time column is provided. 
    
    By default, it looks for an "edges" table for edge data
    and creates a graph from the edges with optional edge attributes corresponding to other 
    columns (except source, target, and time) in the table.
    Additionally, if a node table is specified, node attributes are read from that table
    and added to the graph.

    Args:
        db_path: Path to the SQL database file.
        edge_table: Name of the table containing edges and optional edge attributes. Defaults to "edges".
        node_table: Name of the table containing nodes and optional node attributes. If None, nodes are inferred from edges. Defaults to None.
        source_name: Name of the column representing source nodes in the edge table. Defaults to "source".
        target_name: Name of the column representing target nodes in the edge table. Defaults to "target".
        time_name: Name of the column representing timestamps in the edge table. If None, edges are considered static. Defaults to None.
        node_name: Name of the column representing node IDs in the node table. Defaults to "node_id".
        timestamp_format: Format of the timestamps if time_name is provided. Defaults to "%Y-%m-%d %H:%M:%S".
        time_rescale: Factor to rescale time values (e.g., to convert microseconds to seconds). Defaults to 1.

    Returns:
        Graph: The [graph][pathpyG.Graph] read from the database or the [temporal graph][pathpyG.TemporalGraph] if time_name is provided.
    """
    conn = sqlite3.connect(db_path)

    # Read edges
    edge_query = f"SELECT * FROM {edge_table}"
    edges_df = pd.read_sql_query(edge_query, conn).rename(columns={source_name: "v", target_name: "w"})

    # Create graph
    g: Graph
    if time_name and time_name in edges_df.columns:
        edges_df = edges_df.rename(columns={time_name: "t"})
        g = df_to_temporal_graph(df=edges_df, timestamp_format=timestamp_format, time_rescale=time_rescale)
    else:
        if time_name:
            logger.warning(f"Column '{time_name}' not found in edge table. Reading as static graph.")
        g = df_to_graph(df=edges_df)

    # Read and add node attributes if node_table is provided
    if node_table:
        node_query = f"SELECT * FROM {node_table}"
        nodes_df = pd.read_sql_query(node_query, conn).rename(columns={node_name: "v"})
        add_node_attributes(df=nodes_df, g=g)

    conn.close()

    return g


def write_sql(g: Graph, db_path: str | Path, edge_table: str = "edges", node_table: str = "nodes") -> None:
    """Write a graph to an SQL database file.

    The function writes edge and node data from a [graph][pathpyG.Graph] or a [temporal graph][pathpyG.TemporalGraph]
    to specified tables in the database. By default, it writes to "edges" and "nodes" tables,
    storing edges and edge attributes, as well as nodes and node attributes, respectively.
    For a [temporal graph][pathpyG.TemporalGraph], the time attribute is also included in the edges table.

    Args:
        g: The [graph][pathpyG.Graph] to write to the database.
        db_path: Path to the SQL database file.
        edge_table: Name of the table to store edges and edge attributes. Defaults to "edges".
        node_table: Name of the table to store nodes and node attributes. Defaults to "nodes".
    """
    if isinstance(db_path, str):
        db_path = Path(db_path)
    if db_path.exists():
        logger.warning(f"Database file {db_path} already exists and will be overwritten.")

    conn = sqlite3.connect(db_path)

    # Write edges
    if isinstance(g, TemporalGraph):
        edges_df = temporal_graph_to_df(g)
    else:
        edges_df = graph_to_df(g)
    edges_df.to_sql(edge_table, conn, if_exists="replace", index=False)

    # Write nodes
    nodes_df = pd.DataFrame({"v": list(g.nodes)})
    for attr_name in g.node_attrs():
        nodes_df[attr_name] = g.data[attr_name]
    nodes_df.to_sql(node_table, conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()
