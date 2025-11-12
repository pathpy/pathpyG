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
    """Test the extraction of node attributes."""
    g = read_netzschleuder_graph("karate", "77")
    assert "node__pos" in g.node_attrs()
    assert "node_name" in g.node_attrs()
    assert "node_groups" in g.node_attrs()


def test_edge_attrs():
    """Test the extraction of edge attributes."""
    # Original edge list:
    # source  target  weight
    # 0       1       1
    # 0       2       1
    # 0       3       1
    # 0       8       1
    # 0       9       1
    # 0      11       1
    # 1       2       1
    # 1       8       1
    # 1      11       1
    # 2       4       3
    # 2       5       3
    # 2       8       1
    # 2       9       1
    # 2      11       1
    # 4       5       3
    # 4      14       1
    # 5      14       2
    # 8      11       1
    # 11      12       3

    g = read_netzschleuder_graph("ambassador", "1985_1989", multiedges=True)
    assert "edge_weight" in g.edge_attrs()
    sources = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 5, 8, 11])
    targets = torch.tensor([1, 2, 3, 8, 9, 11, 2, 8, 11, 4, 5, 8, 9, 11, 5, 14, 14, 11, 12])
    weights = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 3, 1, 2, 1, 3])

    for source, target, weight in zip(sources, targets, weights):
        edge = g.mapping.to_idxs([source, target])
        # find edge in edge_index tensor
        mask = (g.data.edge_index[0] == edge[0]) & (g.data.edge_index[1] == edge[1])
        assert (g.data.edge_weight[mask] == weight).all()


def test_graph_attrs():
    """Test the extraction of graph attributes."""
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
