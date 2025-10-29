"""Unit tests for the layout module in pathpyG.visualisations."""

import numpy as np
import pytest

from pathpyG.core.graph import Graph
from pathpyG.visualisations.layout import layout


class TestLayoutAlgorithms:
    """We assume that the underlying algorithms in networkx are tested so we only test the interface."""

    def setup_method(self):
        # Simple triangle graph
        self.g = Graph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])

    @pytest.mark.parametrize(
        "algo",
        [
            "spring",
            "fruchterman-reingold",
            "fr",
            "kamada-kawai",
            "kk",
            "kamada",
            "forceatlas2",
            "fa2",
            "force-atlas2",
            "circular",
            "circle",
            "ring",
            "shell",
            "concentric",
            "grid",
            "lattice-2d",
            "spectral",
            "eigen",
            "random",
            "rand",
        ],
    )
    def test_supported_algorithms(self, algo):
        pos = layout(self.g, layout=algo)
        assert isinstance(pos, dict)
        assert set(pos.keys()) == set(self.g.nodes)
        for coords in pos.values():
            arr = np.array(coords)
            assert arr.shape == (2,)
            assert np.isfinite(arr).all()

    def test_grid_layout_positions(self):
        g = Graph.from_edge_list([("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")])
        pos = layout(g, layout="grid")
        assert isinstance(pos, dict)
        assert set(pos.keys()) == set(g.nodes)
        coords = np.array(list(pos.values()))
        # Should be on a grid: unique x and y values
        assert len(np.unique(coords[:, 0])) > 1
        assert len(np.unique(coords[:, 1])) > 1

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="not recognized"):
            layout(self.g, layout="not-a-layout")

    def test_weight_attribute_missing_raises(self):
        with pytest.raises(ValueError, match="not found"):
            layout(self.g, layout="spring", weight="nonexistent_weight")

    def test_weight_iterable_length_mismatch_raises(self):
        # Too short
        with pytest.raises(ValueError, match="does not match"):
            layout(self.g, layout="spring", weight=[1.0])

    def test_custom_parameters_passed(self):
        pos = layout(self.g, layout="spring", k=0.1, iterations=10)
        assert isinstance(pos, dict)
        assert set(pos.keys()) == set(self.g.nodes)

    def test_weight_as_iterable(self):
        # Correct length
        n_edges = self.g.data.edge_index.size(1)
        pos = layout(self.g, layout="spring", weight=[1.0] * n_edges)
        assert isinstance(pos, dict)
        assert set(pos.keys()) == set(self.g.nodes)
