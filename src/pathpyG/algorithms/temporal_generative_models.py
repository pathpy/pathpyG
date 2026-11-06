"""Generative models for temporal graphs with planted higher-order structure.

The generators here plant structure in the *causal* topology of a temporal graph, i.e. in which
nodes can influence which others along time-respecting paths, rather than in the topology of its
time-aggregated graph. That is the regime in which higher-order models are needed and first-order
models provably fail, which makes these graphs the natural fixtures for
[model selection][pathpyG.core.multi_order_model.MultiOrderModel.estimate_order] and for causality-aware graph learning.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import cumsum

from pathpyG.core.index_map import IndexMap
from pathpyG.core.temporal_graph import TemporalGraph

logger = logging.getLogger("root")


def _generator(seed: Optional[int]) -> torch.Generator:
    """Return a torch random generator, seeded when a seed is given."""
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    return generator


def _cluster_assignment(num_nodes: int, num_clusters: int) -> torch.Tensor:
    """Assign nodes to equally sized clusters of contiguous indices."""
    return torch.arange(num_nodes) * num_clusters // num_nodes


def _temporal_graph_from_walks(
    walks: torch.Tensor, cluster: torch.Tensor, walk_separation: int, num_nodes: int
) -> TemporalGraph:
    """Turn walks given as node sequences into a temporal graph with one event per step.

    The events of a walk occupy consecutive time steps, and consecutive walks are `walk_separation`
    time steps apart, so a separation above `delta` keeps walks from chaining into one another.
    """
    num_walks, num_walk_nodes = walks.shape
    num_events_per_walk = num_walk_nodes - 1

    sources = walks[:, :-1].reshape(-1)
    destinations = walks[:, 1:].reshape(-1)
    walk_start = torch.arange(num_walks) * (num_events_per_walk - 1 + walk_separation)
    time = (walk_start.unsqueeze(1) + torch.arange(num_events_per_walk).unsqueeze(0)).reshape(-1)

    data = Data(
        edge_index=torch.stack([sources, destinations]),
        time=time,
        num_nodes=num_nodes,
    )
    graph = TemporalGraph(data, mapping=IndexMap(np.arange(num_nodes)))
    graph.data.node_cluster = cluster
    return graph


def temporal_clusters(
    num_nodes: int = 30,
    num_edges: int = 560,
    num_clusters: int = 3,
    num_walks: int = 30000,
    order: int = 2,
    cluster_strength: float = 1.0,
    seed: Optional[int] = None,
) -> TemporalGraph:
    r"""Generate a temporal graph whose *causal* topology has clusters but whose static topology does not.

    The construction follows Qarkaxhija, Perri and Scholtes[^1]. A random directed graph is drawn and
    its nodes are split into equally sized clusters. `num_walks` causal walks of `order` adjacent
    time-stamped edges are sampled from that graph, and the time stamps of walks sharing a middle
    node are then exchanged so that a walk is likely to end in the cluster it started in.

    Exchanging time stamps between walks through the same node leaves every time-stamped edge in
    place, so the **time-aggregated weighted graph is preserved exactly**. A first-order model
    therefore cannot see the planted structure at all, while a model of order `order` can:

    $$
    P\big(C(v_{\text{order}}) = C(v_0)\big) \gg \tfrac{1}{\text{num\_clusters}}
    \quad\text{while}\quad
    P\big(C(v_1) = C(v_0)\big) = \tfrac{1}{\text{num\_clusters}}.
    $$

    Info:
        The defaults reproduce the `temporal_clusters` dataset used in the pathpyG tutorials: 30
        nodes in 3 clusters, 560 edges and 30 000 causal walks of two events, i.e. 60 000 events.
        The published dataset was generated with a partial rewiring and reaches
        $P(C(v_2) = C(v_0)) \approx 0.77$; `cluster_strength=1.0` rewires every walk and gives the
        strongest signal the topology allows.

    ??? reference
        [^1]: *L Qarkaxhija, V Perri, I Scholtes: De Bruijn goes Neural: Causality-Aware Graph Neural
            Networks for Time Series Data on Dynamic Graphs, In Proc. of the First Learning on Graphs
            Conference (LoG 2022). [arXiv:2209.08311](https://arxiv.org/abs/2209.08311)*

    Args:
        num_nodes: Number of nodes.
        num_edges: Number of distinct directed edges of the underlying random graph.
        num_clusters: Number of equally sized clusters. Must divide `num_nodes`.
        num_walks: Number of causal walks to generate. Each contributes `order` events.
        order: Number of events of each causal walk, i.e. the order at which the planted structure
            becomes visible.
        cluster_strength: Fraction of walks whose end is rewired towards the start's cluster, between
            zero (no planted structure) and one (as much as the topology allows).
        seed: Seed for reproducible output.

    Returns:
        TemporalGraph: The temporal graph, carrying the cluster of each node as `node_cluster`.

    Raises:
        ValueError: If the parameters are inconsistent, e.g. more edges than node pairs.

    Examples:
        >>> from pathpyG.algorithms.temporal_generative_models import temporal_clusters
        >>> graph = temporal_clusters(num_nodes=12, num_edges=60, num_clusters=3, num_walks=200, seed=0)
        >>> print(graph.m, graph.data.node_cluster.tolist().count(0))
        400 4
    """
    if num_clusters < 1 or num_nodes % num_clusters != 0:
        logger.error("num_clusters %s does not divide num_nodes %s", num_clusters, num_nodes)
        raise ValueError(f"num_clusters ({num_clusters}) must be at least one and divide num_nodes ({num_nodes})")
    if num_edges > num_nodes * num_nodes:
        logger.error("num_edges %s exceeds the %s possible node pairs", num_edges, num_nodes * num_nodes)
        raise ValueError(f"num_edges ({num_edges}) cannot exceed the {num_nodes * num_nodes} possible node pairs")
    if order < 1:
        logger.error("order must be at least one, got %s", order)
        raise ValueError(f"order must be at least one, got {order}")
    if not 0.0 <= cluster_strength <= 1.0:
        logger.error("cluster_strength must lie between zero and one, got %s", cluster_strength)
        raise ValueError(f"cluster_strength must lie between zero and one, got {cluster_strength}")

    generator = _generator(seed)
    cluster = _cluster_assignment(num_nodes, num_clusters)

    # A directed random graph, drawn as distinct node pairs so that no edge is duplicated.
    chosen = torch.randperm(num_nodes * num_nodes, generator=generator)[:num_edges]
    chosen, _ = torch.sort(chosen)
    edge_sources, edge_destinations = chosen // num_nodes, chosen % num_nodes

    out_degree = torch.bincount(edge_sources, minlength=num_nodes)
    if bool((out_degree == 0).any()) and order > 1:
        logger.warning(
            "%s node(s) have no outgoing edge; walks are sampled among the nodes that do",
            int((out_degree == 0).sum().item()),
        )
    row_pointer = cumsum(out_degree)

    # Sample walks: a uniformly random edge, then a uniformly random continuation at each step.
    walks = torch.empty((num_walks, order + 1), dtype=torch.long)
    first_edge = torch.randint(num_edges, (num_walks,), generator=generator)
    walks[:, 0] = edge_sources[first_edge]
    walks[:, 1] = edge_destinations[first_edge]
    for step in range(2, order + 1):
        current = walks[:, step - 1]
        degree = out_degree[current]
        if bool((degree == 0).any()):
            logger.error("sampling reached a node without outgoing edges at step %s", step)
            raise ValueError(
                "the random graph contains nodes without outgoing edges that a walk can reach. "
                "Increase num_edges or decrease order."
            )
        offset = (torch.rand(num_walks, generator=generator) * degree).to(torch.long).clamp(max=degree - 1)
        walks[:, step] = edge_destinations[row_pointer[current] + offset]

    if cluster_strength > 0 and order >= 1:
        walks = _rewire_walk_ends(walks, cluster, cluster_strength, generator)

    return _temporal_graph_from_walks(walks, cluster, walk_separation=1, num_nodes=num_nodes)


def _rewire_walk_ends(
    walks: torch.Tensor, cluster: torch.Tensor, cluster_strength: float, generator: torch.Generator
) -> torch.Tensor:
    """Exchange the last node between walks that share their second-to-last node.

    Walks are grouped by their second-to-last node; inside each group the walks are ordered by the
    cluster they start in and the available last nodes by the cluster they lie in, and the two are
    paired off positionally. That matches as many walks as the group allows to an ending in their
    own cluster while leaving every time-stamped edge of the graph in place.
    """
    num_walks = walks.size(0)
    selected = torch.randperm(num_walks, generator=generator)[: int(round(cluster_strength * num_walks))]
    if selected.numel() == 0:
        return walks

    penultimate = walks[selected, -2]
    last = walks[selected, -1]

    # Lexicographic sort by (penultimate node, cluster): sorting on the minor key first and then
    # stably on the major key leaves the groups intact and ordered by cluster within each group.
    def grouped_order(cluster_key: torch.Tensor) -> torch.Tensor:
        by_cluster = torch.argsort(cluster_key, stable=True)
        return by_cluster[torch.argsort(penultimate[by_cluster], stable=True)]

    walk_order = grouped_order(cluster[walks[selected, 0]])
    ending_order = grouped_order(cluster[last])

    rewired = walks.clone()
    rewired[selected[walk_order], -1] = last[ending_order]
    return rewired


def higher_order_block_walks(
    num_nodes: int,
    num_clusters: int,
    num_walks: int,
    order: int = 2,
    walk_length: Optional[int] = None,
    intra_cluster_probability: float = 0.8,
    walk_separation: int = 1,
    seed: Optional[int] = None,
) -> TemporalGraph:
    r"""Generate time-respecting walks whose Markov order is exactly `order`, by construction.

    Each walk $v_0 \to \dots \to v_L$ is drawn with a *lagged* dependency: the first `order` nodes
    are uniform over all nodes, and from then on

    $$
    v_i \sim \begin{cases}
        \mathrm{Uniform}\big(C(v_{i-\text{order}})\big) & \text{with probability } p, \\
        \mathrm{Uniform}(V) & \text{otherwise},
    \end{cases}
    $$

    so $v_i$ depends on the node `order` steps back and on nothing in between. Every marginal of
    order below `order` is therefore exactly uniform, which makes the true Markov order exactly
    `order` and gives model selection an unambiguous ground truth to be tested against.

    Unlike [`temporal_clusters`][pathpyG.algorithms.temporal_clusters], this does not preserve any
    given first-order topology: it samples the topology and the higher-order structure together.

    Args:
        num_nodes: Number of nodes.
        num_clusters: Number of equally sized clusters. Must divide `num_nodes`.
        num_walks: Number of walks to generate.
        order: The Markov order of the generated process.
        walk_length: Number of events of each walk. Defaults to `order` and must be at least `order`.
        intra_cluster_probability: Probability $p$ of drawing a node from the cluster of the node
            `order` steps back rather than uniformly.
        walk_separation: Time steps between the last event of a walk and the first of the next.
            Set it above `delta` to keep walks from chaining into one another.
        seed: Seed for reproducible output.

    Returns:
        TemporalGraph: The temporal graph, carrying the cluster of each node as `node_cluster`.

    Examples:
        >>> from pathpyG.algorithms.temporal_generative_models import higher_order_block_walks
        >>> graph = higher_order_block_walks(
        ...     num_nodes=12, num_clusters=3, num_walks=100, order=2, seed=0
        ... )
        >>> print(graph.m)
        200
    """
    if num_clusters < 1 or num_nodes % num_clusters != 0:
        logger.error("num_clusters %s does not divide num_nodes %s", num_clusters, num_nodes)
        raise ValueError(f"num_clusters ({num_clusters}) must be at least one and divide num_nodes ({num_nodes})")
    if order < 1:
        logger.error("order must be at least one, got %s", order)
        raise ValueError(f"order must be at least one, got {order}")
    if walk_length is None:
        walk_length = order
    if walk_length < order:
        logger.error("walk_length %s is smaller than order %s", walk_length, order)
        raise ValueError(f"walk_length ({walk_length}) must be at least order ({order})")
    if not 0.0 <= intra_cluster_probability <= 1.0:
        logger.error("intra_cluster_probability must lie between zero and one, got %s", intra_cluster_probability)
        raise ValueError(f"intra_cluster_probability must lie between zero and one, got {intra_cluster_probability}")

    generator = _generator(seed)
    cluster = _cluster_assignment(num_nodes, num_clusters)
    cluster_size = num_nodes // num_clusters

    walks = torch.empty((num_walks, walk_length + 1), dtype=torch.long)
    walks[:, :order] = torch.randint(num_nodes, (num_walks, order), generator=generator)
    for step in range(order, walk_length + 1):
        lagged = walks[:, step - order]
        within_cluster = cluster[lagged] * cluster_size + torch.randint(cluster_size, (num_walks,), generator=generator)
        anywhere = torch.randint(num_nodes, (num_walks,), generator=generator)
        follow_cluster = torch.rand(num_walks, generator=generator) < intra_cluster_probability
        walks[:, step] = torch.where(follow_cluster, within_cluster, anywhere)

    return _temporal_graph_from_walks(walks, cluster, walk_separation=walk_separation, num_nodes=num_nodes)
