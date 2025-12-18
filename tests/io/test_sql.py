"""Unit tests for sql.py module."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pathpyG.core.graph import Graph
from pathpyG.io.sql import read_sql, write_sql


class TestReadSql:
    """Tests for read_sql function."""

    def test_read_sql_static_graph(self, tmp_path):
        """Test reading a static graph from SQL database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        # Create test edge table
        edges_data = {"source": ["A", "B", "C"], "target": ["B", "C", "A"], "weight": [1.0, 2.0, 3.0]}
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_sql("edges", conn, if_exists="replace", index=False)
        conn.close()

        # Read graph
        g = read_sql(str(db_path))

        assert isinstance(g, Graph)
        assert len(g.nodes) == 3
        assert len(g.edges) == 3
        assert g.data["edge_weight"].tolist() == [1.0, 2.0, 3.0]

    def test_read_sql_temporal_graph(self, tmp_path):
        """Test reading a temporal graph from SQL database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        # Create test edge table with timestamps
        edges_data = {
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"],
            "timestamp": ["2023-01-01 10:00:00", "2023-01-01 10:00:10", "2023-01-01 10:00:25"],
        }
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_sql("edges", conn, if_exists="replace", index=False)
        conn.close()

        # Read temporal graph
        g = read_sql(str(db_path), time_name="timestamp")

        assert len(g.nodes) == 3
        assert len(g.edges) == 3
        assert "time" in g.data
        assert g.data["time"].tolist() == [0, 10, 25]

    def test_read_sql_with_node_table(self, tmp_path):
        """Test reading graph with node attributes from separate node table."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        # Create edge table
        edges_data = {"source": ["A", "B"], "target": ["B", "C"]}
        pd.DataFrame(edges_data).to_sql("edges", conn, if_exists="replace", index=False)

        # Create node table with attributes
        nodes_data = {
            "node_id": ["A", "B", "C"],
            "label": ["Node A", "Node B", "Node C"],
            "color": ["red", "blue", "green"],
        }
        pd.DataFrame(nodes_data).to_sql("nodes", conn, if_exists="replace", index=False)
        conn.close()

        # Read graph
        g = read_sql(str(db_path), node_table="nodes")

        assert g.nodes == ["A", "B", "C"]
        assert g.data["node_label"].tolist() == ["Node A", "Node B", "Node C"]
        assert g.data["node_color"].tolist() == ["red", "blue", "green"]
        assert g.data.edge_index.shape[1] == 2

    def test_read_sql_custom_column_names(self, tmp_path):
        """Test reading graph with custom column names."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        # Create edge table with custom column names
        edges_data = {"from_node": ["X", "Y"], "to_node": ["Y", "Z"]}
        pd.DataFrame(edges_data).to_sql("my_edges", conn, if_exists="replace", index=False)
        conn.close()

        # Read graph with custom names
        g = read_sql(str(db_path), edge_table="my_edges", source_name="from_node", target_name="to_node")

        assert len(g.nodes) == 3
        assert len(g.edges) == 2

    def test_read_sql_missing_time_column(self, tmp_path):
        """Test reading graph when time_name column doesn't exist."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        edges_data = {"source": ["A", "B"], "target": ["B", "C"]}
        pd.DataFrame(edges_data).to_sql("edges", conn, if_exists="replace", index=False)
        conn.close()

        # Should return static graph even if time_name is specified but not present and log a warning
        with patch("pathpyG.io.sql.logger") as mock_logger:
            g = read_sql(str(db_path), time_name="nonexistent_time")
            mock_logger.warning.assert_called_once()

        assert isinstance(g, Graph)
        assert len(g.edges) == 2


class TestWriteSql:
    """Tests for write_sql function."""

    def test_write_sql_creates_database(self, tmp_path):
        """Test that write_sql creates a database file."""
        db_path = tmp_path / "output.db"

        # Create a simple graph
        g = Graph.from_edge_list([("A", "B"), ("B", "C")])

        # Write to database
        write_sql(g, db_path)

        assert db_path.exists()

    def test_write_sql_creates_tables(self, tmp_path):
        """Test that write_sql creates edge and node tables."""
        db_path = tmp_path / "output.db"

        g = Graph.from_edge_list([("A", "B"), ("B", "C")])
        write_sql(g, db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert "edges" in tables
        assert "nodes" in tables
        conn.close()

    def test_write_sql_custom_table_names(self, tmp_path):
        """Test write_sql with custom table names."""
        db_path = tmp_path / "output.db"

        g = Graph.from_edge_list([("A", "B")])

        write_sql(g, db_path, edge_table="my_edges", node_table="my_nodes")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert "my_edges" in tables
        assert "my_nodes" in tables
        conn.close()

    def test_write_sql_overwrite_warning(self, tmp_path):
        """Test that write_sql logs warning when overwriting existing database."""
        db_path = tmp_path / "output.db"

        g = Graph.from_edge_list([("A", "B")])

        # Write once
        write_sql(g, db_path)

        # Write again and check for warning
        with patch("pathpyG.io.sql.logger") as mock_logger:
            write_sql(g, db_path)
            mock_logger.warning.assert_called_once()

    def test_write_read_roundtrip(self, tmp_path):
        """Test that graph can be written and read back."""
        db_path = tmp_path / "roundtrip.db"

        # Create and write graph
        g_original = Graph.from_edge_list([("A", "B"), ("B", "C")])
        g_original.data["edge_weight"] = [1.0, 2.0]
        g_original.data["node_label"] = ["Node A", "Node B", "Node C"]

        write_sql(g_original, db_path)

        # Read back
        g_read = read_sql(str(db_path), node_table="nodes")

        assert len(g_read.nodes) == len(g_original.nodes)
        assert len(g_read.edges) == len(g_original.edges)
        assert g_read.data["edge_weight"].tolist() == g_original.data["edge_weight"]
        assert g_read.data["node_label"].tolist() == g_original.data["node_label"]


class TestReadWriteIntegration:
    """Integration tests for read and write operations."""

    def test_connection_closed_after_read(self, tmp_path):
        """Test that database connection is properly closed after read."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        edges_data = {"source": ["A"], "target": ["B"]}
        pd.DataFrame(edges_data).to_sql("edges", conn, if_exists="replace", index=False)
        conn.close()

        # Read graph
        read_sql(str(db_path))

        # Should be able to delete file if connection is closed
        Path(db_path).unlink()
        assert not Path(db_path).exists()
