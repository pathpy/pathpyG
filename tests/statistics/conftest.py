import pytest

from pathpyG.core.graph import Graph


@pytest.fixture
def simple_graph() -> Graph:
    """Return a simple undirected graph."""
    return Graph.from_edge_list(
        [
            ("a", "b"),
            ("b", "a"),
            ("b", "c"),
            ("c", "b"),
            ("b", "d"),
            ("d", "b"),
            ("d", "e"),
            ("e", "d"),
            ("c", "e"),
            ("e", "c"),
        ],
        is_undirected=True,
    )


@pytest.fixture
def toy_example_graph() -> Graph:
    """Return an undirected toy example graph."""
    return Graph.from_edge_list(
        [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f"), ("f", "g"), ("g", "d"), ("d", "f"), ("b", "d")]
    ).to_undirected()


@pytest.fixture
def toy_example_graph_directed() -> Graph:
    """Return a directed toy example graph."""
    return Graph.from_edge_list(
        [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f"), ("f", "g"), ("g", "d"), ("d", "f"), ("b", "d")]
    )
