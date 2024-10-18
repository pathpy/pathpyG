from __future__ import annotations

import torch

from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.algorithms.rolling_time_window import RollingTimeWindow


def test_rolling_time_window(long_temporal_graph):
    r = RollingTimeWindow(long_temporal_graph, 10, 10, False)
    snapshots = []
    for g in r:
        snapshots.append(g)
    # aggregate network from 1 to 10
    assert snapshots[0].N == 5 and snapshots[0].M == 4
    # aggregate network from 11 to 20
    assert snapshots[1].N == 7 and snapshots[1].M == 3
    # aggregate network from 21 to 30
    assert snapshots[2].N == 8 and snapshots[2].M == 6
    # aggregate network from 31 to 40
    assert snapshots[3].N == 8 and snapshots[3].M == 3
    # aggregate network from 41 to 50
    assert snapshots[4].N == 9 and snapshots[4].M == 4
