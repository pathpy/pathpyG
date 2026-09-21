"""This module tests the resolution of the example datasets shipped with the tutorials."""

# pylint: disable=missing-function-docstring

import io
from pathlib import Path

from pathpyG.io.datasets import EXAMPLE_DATA_URL, example_data
from pathpyG.io.pandas import read_csv_path_data


def test_example_data_finds_local_copy():
    # every dataset in docs/data resolves to that folder, independent of the working directory
    resolved = Path(example_data("temporal_clusters.tedges"))

    assert resolved.is_file()
    assert resolved.parent.name == "data" and resolved.parent.parent.name == "docs"


def test_example_data_falls_back_to_url():
    assert example_data("no_such_dataset.tedges") == EXAMPLE_DATA_URL + "no_such_dataset.tedges"


def test_read_csv_path_data_from_buffer():
    buf = io.StringIO("a,b,c,2\na,b,4\n")
    paths = read_csv_path_data(path_or_buf=buf, sep=",", weight=True)

    assert paths.num_paths == 2
    assert paths.data.dag_weight.sum().item() == 6.0
    assert paths.get_walk(0) == ("a", "b", "c")
