"""Module for node similarity measures in graphs."""

import numpy as _np
import scipy as _sp

from pathpyG.algorithms.shortest_paths import shortest_paths_dijkstra
from pathpyG.core.graph import Graph
from pathpyG.statistics.degrees import degree_sequence


def inverse_path_length(graph: Graph, v: str | int, w: str | int) -> float:
    r"""Compute the inverse path length similarity between two nodes.

    Given a [graph][pathpyG.core.graph.Graph] $G=(V, E)$ and two nodes $v, w \in V$,
    the inverse path length similarity is defined as

    $$
    S_{IPL}(v, w) = \begin{cases} \frac{1}{d(v, w)} & \text{if } d(v, w) > 0 \\ \infty & \text{if } d(v, w) = 0 \end{cases}
    $$

    with the shortest path length $d(v, w)$ between nodes $v$ and $w$.

    Args:
        graph: The input graph.
        v: The first node given the node label (or index if no label is provided).
        w: The second node given the node label (or index if no label is provided).

    Returns:
        The inverse path length similarity between nodes `v` and `w`.
    """
    dist, _ = shortest_paths_dijkstra(graph)
    d = dist[graph.mapping.to_idx(v), graph.mapping.to_idx(w)]
    if d == 0:
        return _np.inf
    else:
        return 1 / d  # type: ignore[return-value]


def common_neighbors(graph: Graph, v: str | int, w: str | int) -> float:
    r"""Compute the number of common neighbors between two nodes.

    $$
    S_{CN}(v, w) = |\mathcal{N}(v) \cap \mathcal{N}(w)|
    $$

    Args:
        graph: The input graph.
        v: The first node given the node label (or index if no label is provided).
        w: The second node given the node label (or index if no label is provided).

    Returns:
        The number of common neighbors between nodes `v` and `w`.
    """
    N_v = set([x for x in graph.successors(v)])
    N_w = set([x for x in graph.successors(w)])
    return len(N_v.intersection(N_w))


def overlap_coefficient(graph: Graph, v: str | int, w: str | int) -> float:
    r"""Compute the overlap coefficient between two nodes.

    The overlap coefficient between two nodes $v, w \in V$ is defined as

    $$
    S_{OC}(v, w) = \frac{|\mathcal{N}(v) \cap \mathcal{N}(w)|}{\min(|\mathcal{N}(v)|, |\mathcal{N}(w)|)}
    $$

    where $\mathcal{N}(v)$ and $\mathcal{N}(w)$ are the sets of neighbors of nodes $v$ and $w$, respectively.

    Args:
        graph: The input graph.
        v: The first node given the node label (or index if no label is provided).
        w: The second node given the node label (or index if no label is provided).

    Returns:
        The overlap coefficient between nodes `v` and `w`.
    """
    N_v = set([x for x in graph.successors(v)])
    N_w = set([x for x in graph.successors(w)])
    return len(N_v.intersection(N_w)) / min(len(N_v), len(N_w))


def jaccard_similarity(graph: Graph, v: str | int, w: str | int) -> float:
    r"""Compute the Jaccard similarity between two nodes.

    The Jaccard similarity between two nodes $v, w \in V$ is defined as

    $$
    S_{J}(v, w) = \frac{|\mathcal{N}(v) \cap \mathcal{N}(w)|}{|\mathcal{N}(v) \cup \mathcal{N}(w)|}
    $$

    where $\mathcal{N}(v)$ and $\mathcal{N}(w)$ are the sets of neighbors of nodes $v$ and $w$, respectively.

    ??? reference
        For more details, see Equation (7.38) in *Networks*[^1] by Mark Newman.

    [^1]: *Newman, M. E. J. Networks. (Oxford University Press, 2018). doi:10.1093/oso/9780198805090.001.0001.*

    Args:
        graph: The input graph.
        v: The first node given the node label (or index if no label is provided).
        w: The second node given the node label (or index if no label is provided).

    Returns:
        The Jaccard similarity between nodes `v` and `w`.
    """
    N_v = set([x for x in graph.successors(v)])
    N_w = set([x for x in graph.successors(w)])
    intersection = N_v.intersection(N_w)
    if len(N_v) == 0 and len(N_w) == 0:
        return 1
    else:
        return len(intersection) / (len(N_v) + len(N_w) - len(intersection))


def adamic_adar_index(graph: Graph, v: str | int, w: str | int) -> float:
    r"""Compute the Adamic-Adar index between two nodes.

    The Adamic-Adar index between two nodes $v, w \in V$ is defined as

    $$
    S_{AA}(v, w) = \sum_{u \in \mathcal{N}(v) \cap \mathcal{N}(w)} \frac{1}{\log(|\mathcal{N}(u)|)}
    $$

    where $\mathcal{N}(v)$ and $\mathcal{N}(w)$ are the sets of neighbors of nodes $v$ and $w$, respectively.

    ??? reference
        Proposed by Adamic and Adar in "Friends and neighbors on the web"[^1].

    [^1]: *Adamic, L. A. & Adar, E. Friends and neighbors on the Web. Social Networks 25, 211-230 (2003).*

    Args:
        graph: The input graph.
        v: The first node given the node label (or index if no label is provided).
        w: The second node given the node label (or index if no label is provided).

    Returns:
        The Adamic-Adar index between nodes `v` and `w`.
    """
    A = 0
    N_v = set([x for x in graph.successors(v)])
    N_w = set([x for x in graph.successors(w)])
    for u in N_v.intersection(N_w):
        A += 1 / _np.log(graph.out_degrees[u])
    return A


def cosine_similarity(graph: Graph, v: str | int, w: str | int) -> float:
    r"""Compute the cosine similarity between two nodes.

    The cosine similarity between two nodes $v, w \in V$ is defined as

    $$
    S_{COS}(v, w) = \frac{\mathbf{A}_v \cdot \mathbf{A}_w}{\|\mathbf{A}_v\|_2 \|\mathbf{A}_w\|_2}
    $$

    where $\mathbf{A}_v$ and $\mathbf{A}_w$ are row vectors of the adjacency matrix corresponding to nodes $v$ and $w$, respectively, $\|\cdot\|_2$ denotes the Euclidean norm, and $\cdot$ denotes the dot product.

    ??? reference
        For more details, see Equation (7.35) in *Networks*[^1] by Mark Newman.

    [^1]: *Newman, M. E. J. Networks. (Oxford University Press, 2018). doi:10.1093/oso/9780198805090.001.0001.*

    Args:
        graph: The input graph.
        v: The first node given the node label (or index if no label is provided).
        w: The second node given the node label (or index if no label is provided).

    Returns:
        The cosine similarity between nodes `v` and `w`.
    """
    if graph.degrees()[v] == 0 or graph.degrees()[w] == 0:  # type: ignore[index]
        return 0
    else:
        A = graph.sparse_adj_matrix().todense()
        v_v = A[graph.mapping.to_idx(v)].A1
        v_w = A[graph.mapping.to_idx(w)].A1
        return _np.dot(v_v, v_w) / (_np.linalg.norm(v_v) * _np.linalg.norm(v_w))


def katz_index(graph: Graph, v: str | int, w: str | int, beta: float) -> float:
    r"""Compute the Katz index between two nodes.

    The Katz index for all pairs of nodes $V^2$ is defined as

    $$
    S_{Katz} = \sum_{l=1}^{\infty} \beta^l A^l = (I - \beta A)^{-1} - I
    $$

    where $A$ is the adjacency matrix of the graph, $I$ is the identity matrix, and $\beta$ is a parameter controlling the weight of longer paths.
    To get $S_{Katz}(v, w)$, the entry corresponding to nodes $v$ and $w$ is selected from the matrix $S_{Katz}$.

    ??? reference
        While the Katz index was originally proposed by Leo Katz in 1953[^1] as a measure of social influence of a node (centrality),
        it can also be used as a node similarity measure between two nodes.

    [^1]: *Katz, L. A new status index derived from sociometric analysis. Psychometrika 18, 39-43 (1953).*

    Args:
        graph: The input graph.
        v: The first node given the node label (or index if no label is provided).
        w: The second node given the node label (or index if no label is provided).
        beta: Parameter controlling the weight of longer paths.

    Returns:
        The Katz index between nodes `v` and `w`.
    """
    A = graph.sparse_adj_matrix()
    I = _sp.sparse.identity(graph.n)  # noqa: E741
    S = _sp.sparse.linalg.inv(I - beta * A) - I
    return S[graph.mapping.to_idx(v), graph.mapping.to_idx(w)]


def LeichtHolmeNewman_index(graph: Graph, v: str | int, w: str | int, alpha: float) -> float:
    r"""Compute the Leicht-Holme-Newman index between two nodes.
    
    The Leicht-Holme-Newman index for all pairs of nodes $V^2$ is defined as

    $$
    S_{LHN} = 2m \lambda_1 D^{-1} \left(I - \frac{\alpha}{\lambda_1} A\right)^{-1} D^{-1}
    $$

    where [$m$][pathpyG.core.graph.Graph.m] is the number of edges in the graph, $\lambda_1$ is the largest eigenvalue of the adjacency matrix $A$, $D$ is the diagonal degree matrix, and $I$ is the identity matrix.
    To obtain $S_{LHN}(v, w)$, the entry corresponding to nodes $v$ and $w$ is selected from the matrix $S_{LHN}$.

    ??? reference
        Proposed by Leicht, Holme, and Newman in "Vertex similarity in networks"[^1].

    [^1]: *Leicht, E. A., Holme, P. & Newman, M. E. J. Vertex similarity in networks. Phys. Rev. E 73, 026120 (2006).*

    Args:
        graph: The input graph.
        v: The first node given the node label (or index if no label is provided).
        w: The second node given the node label (or index if no label is provided).
        alpha: Parameter controlling the weight of longer paths.

    Returns:
        The Leicht-Holme-Newman index between nodes `v` and `w`.
    """
    A = graph.sparse_adj_matrix()
    ev = _sp.sparse.linalg.eigs(A, which="LM", k=2, return_eigenvectors=False)
    eigenvalues_sorted = _np.sort(_np.absolute(ev))
    m = graph.m
    lambda_1 = eigenvalues_sorted[1]
    D = _sp.sparse.diags(degree_sequence(graph).numpy()).tocsc()
    I = _sp.sparse.identity(graph.n).tocsc()  # noqa: E741
    S = (
        2
        * m
        * lambda_1
        * _sp.sparse.linalg.inv(D)
        * _sp.sparse.linalg.inv(I - alpha * A / lambda_1)
        * _sp.sparse.linalg.inv(D)
    )
    return S[graph.mapping.to_idx(v), graph.mapping.to_idx(w)]
