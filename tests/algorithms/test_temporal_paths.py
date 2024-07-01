from __future__ import annotations

import torch
import numpy as np

from pathpyG.core.graph import Graph
from pathpyG.algorithms.temporal import temporal_shortest_paths


# def test_time_respecting_paths(long_temporal_graph):
#     paths = time_respecting_paths(long_temporal_graph, delta=5)
#     assert paths[1] == [['a', 'b'],
#                         ['b', 'f'],
#                         ['b', 'i'],
#                         ['c', 'f'],
#                         ['c', 'i'],
#                         ['f', 'h']]
#     assert paths[3] == [['a', 'b', 'c', 'd'],
#                         ['a', 'b', 'c', 'e'],
#                         ['c', 'f', 'a', 'g']]
#     assert paths[2] == [['a', 'c', 'h'],
#                         ['a', 'g', 'h']]


# def test_temporal_shortest_paths(long_temporal_graph):
#     sp, sp_lengths, counts = temporal_shortest_paths(long_temporal_graph, delta=5)
#     assert torch.equal(sp[1], torch.tensor([[0, 1],
#         [0, 2],
#         [0, 6],
#         [1, 2],
#         [1, 5],
#         [1, 8],
#         [2, 3],
#         [2, 4],
#         [2, 5],
#         [2, 7],
#         [2, 8],
#         [5, 0],
#         [5, 7],
#         [6, 7],
#         [7, 5],
#         [7, 8],
#         [8, 1]]))
#     assert torch.equal(sp[2], torch.tensor([[0, 2, 7],
#         [0, 6, 7],
#         [1, 2, 3],
#         [1, 2, 4],
#         [2, 5, 0],
#         [5, 0, 6]]))
#     assert torch.equal(sp[3], torch.tensor([[0, 1, 2, 3],
#         [0, 1, 2, 4],
#         [2, 5, 0, 6]]))
