"""This module tests high-level functions of the netzschleuder module."""

import pytest

import torch

from pathpyG import Graph, TemporalGraph
from pathpyG.io import list_netzschleuder_records, read_netzschleuder_graph, read_netzschleuder_record


def test_list_netzschleuder_records():
    """Test the list_netzschleuder_records() function."""

    # Test the function with a valid URL.
    records = list_netzschleuder_records()
    print(records)
    assert len(records) > 0

    # Test the function with an invalid URL.
    url = "https://networks.skewed.de/invalid-url"
    with pytest.raises(Exception, match="Could not connect to netzschleuder repository at"):
        records = list_netzschleuder_records(url)


def test_node_attrs():
    """Test the extraction of node attributes"""
    g = read_netzschleuder_graph("karate", "77")
    assert "node__pos" in g.node_attrs()
    assert "node_name" in g.node_attrs()
    assert "node_groups" in g.node_attrs()


def test_edge_attrs():
    """Test the extraction of edge attributes"""
    g = read_netzschleuder_graph("ambassador", "1985_1989", multiedges=True)
    assert "edge_weight" in g.edge_attrs()
    print(g.data.edge_weight)
    assert torch.equal(
        g.data.edge_weight,
        torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                3,
                1,
                1,
                1,
                3,
                1,
                3,
                3,
                1,
                1,
                3,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                3,
                1,
                3,
                3,
                1,
                2,
            ]
        ),
    )


def test_graph_attrs():
    """Test the extraction of graph attributes"""
    g = read_netzschleuder_graph("karate", "77")
    assert "analyses_diameter" in g.data
    assert g.data.analyses_diameter == 5


def test_read_netzschleuder_record():
    """Test the read_netzschleuder_record() function."""

    # Test the function with a valid URL.
    record_name = list_netzschleuder_records()[0]
    record = read_netzschleuder_record(record_name)
    assert isinstance(record, dict)
    assert record

    # Test the function with an invalid URL.
    url = "https://networks.skewed.de/invalid-url"
    with pytest.raises(Exception, match="Could not connect to netzschleuder repository at"):
        record = read_netzschleuder_record(record_name, url)


def test_read_netzschleuder_graph():
    """Test the read_netzschleuder_graph() function for timestamped data."""

    g = read_netzschleuder_graph(name="email_company")
    assert isinstance(g, Graph)
    assert g.n == 167
    assert g.m == 5784


def test_read_netzschleuder_graph_temporal():
    """Test the read_netzschleuder_graph() function for timestamped data."""

    g = read_netzschleuder_graph(name="email_company", time_attr="time", multiedges=True)
    assert isinstance(g, TemporalGraph)
    assert g.n == 167
    assert g.m == 82927
    assert g.start_time == 1262454010
    assert g.end_time == 1285884492
    assert "edge_weight" in g.edge_attrs()
