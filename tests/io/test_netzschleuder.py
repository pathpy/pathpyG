"""This module tests high-level functions of the netzschleuder module."""

import pytest

from pathpyG import Graph
from pathpyG.io import list_netzschleuder_records, read_netzschleuder_network, read_netzschleuder_record


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


def test_read_netzschleuder_network():
    """Test the read_netzschleuder_network() function."""

    # Test the function with valid URLs.
    network = read_netzschleuder_network(name="7th_graders")
    assert isinstance(network, Graph)
    assert network.N == 29
    assert network.M == 740
    assert network.is_directed() == True

    network = read_netzschleuder_network(name="karate", net="77")
    assert isinstance(network, Graph)
    assert network.N == 34
    assert network.M == 154
    assert network.is_directed() == False

    # Test the function without a network name.
    with pytest.raises(Exception, match="Could not connect to netzschleuder repository at"):
        network = read_netzschleuder_network("karate")
