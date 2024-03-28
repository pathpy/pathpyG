from __future__ import annotations

import torch
import numpy as np

from pathpyG.core.Graph import Graph
from pathpyG.algorithms.temporal import time_respecting_paths, temporal_shortest_paths


def test_time_respecting_paths(long_temporal_graph):
    paths = time_respecting_paths(long_temporal_graph, delta=5)
    assert paths[1] == [['a', 'b'],
                        ['b', 'f'],
                        ['b', 'i'],
                        ['c', 'f'],
                        ['c', 'i'],
                        ['f', 'h'],
                        ['h', 'f'],
                        ['h', 'i'],
                        ['i', 'b']]
    assert paths[3] == [['a', 'b', 'c', 'd'],
                        ['a', 'b', 'c', 'e'],
                        ['c', 'f', 'a', 'g']]
    assert paths[2] == [['a', 'c', 'h'],
                        ['a', 'g', 'h']]


def test_temporal_shortest_paths(long_temporal_graph):
    sp, sp_lengths, counts = temporal_shortest_paths(long_temporal_graph, delta=5)
    assert sp['a']['h'] == {('a', 'c', 'h'), ('a', 'g', 'h')}
    assert sp_lengths['a']['h'] == 2
    assert sp_lengths['a']['i'] == np.inf
