"""Algorithms for temporal path calculation and graph metrics.

The functions and submodules in this module allow to compute
time-respecting or causal paths in temporal graphs and to
calculate (temporal) and higher-order graph metrics like centralities.

Time-respecting walks can be counted without being materialized
([`walk_counts`][pathpyG.algorithms.temporal.walk_counts]), which is what makes multi-order
statistics for temporal graphs tractable, or extracted explicitly
([`extract_time_respecting_walks`][pathpyG.algorithms.temporal.extract_time_respecting_walks] and
[`extract_causal_paths`][pathpyG.algorithms.temporal.extract_causal_paths]).
[`temporal_generative_models`][pathpyG.algorithms.temporal_generative_models] generates temporal
graphs with structure planted in their causal topology.

Example:
    ```py
    # Import pathpyG
    import pathpyG as pp

    # Generate a toy example for a temporal graph.
    g = pp.TemporalGraph.from_edge_list(
        [("b", "c", 2), ("a", "b", 1), ("c", "d", 3), ("d", "a", 4), ("b", "d", 2), ("d", "a", 6), ("a", "b", 7)]
    )

    # Extract DAG capturing causal interaction sequences in temporal graph.
    e_i = EventGraph.build_edge_index(g, delta=1)
    dag = pp.Graph.from_edge_index(e_i)
    print(dag)

    # Calculate shortest time-respecting pathas
    dist, pred = pp.algorithms.temporal.temporal_shortest_paths(g, delta=1)

    # Extract the maximal time-respecting paths themselves
    paths = pp.algorithms.extract_causal_paths(g, delta=1)
    print(paths)
    ```
"""

from pathpyG.algorithms import centrality, generative_models, shortest_paths, temporal_generative_models
from pathpyG.algorithms.components import connected_components, largest_connected_component
from pathpyG.algorithms.rolling_time_window import RollingTimeWindow
from pathpyG.algorithms.temporal import (
    extract_causal_paths,
    extract_time_respecting_walks,
    temporal_shortest_paths,
    walk_counts,
)
from pathpyG.algorithms.temporal_generative_models import higher_order_block_walks, temporal_clusters
from pathpyG.algorithms.weisfeiler_leman import WeisfeilerLeman_test

__all__ = [
    "temporal_shortest_paths",
    "walk_counts",
    "extract_time_respecting_walks",
    "extract_causal_paths",
    "temporal_clusters",
    "higher_order_block_walks",
    "centrality",
    "generative_models",
    "temporal_generative_models",
    "shortest_paths",
    "connected_components",
    "largest_connected_component",
    "RollingTimeWindow",
    "WeisfeilerLeman_test",
]
