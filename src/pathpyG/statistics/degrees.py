"""Module for degree-related statistics of graphs."""

import numpy as _np
import torch

from pathpyG.core.graph import Graph


def degree_sequence(graph: Graph, mode: str = "total") -> torch.Tensor:
    r"""Calculates the (unweighted) degree sequence of a [graph][pathpyG.core.graph.Graph].

    Returns the degree sequence of a [graph][pathpyG.core.graph.Graph] $G=(V, E)$ defined as

    $$
    \left\{d_{mode}(v_1), d_{mode}(v_2), \ldots, d_{mode}(v_n)\right\} \text{ for all nodes } v_i \in V
    $$

    with $d_{mode}(v_i)$ being the degree of node $v_i$ in mode 'in', 'out' or 'total'.
    The modes are defined as follows:

    - 'in': In-degree $d_{in}(v_i)$ of node $v_i$, i.e. the number of incoming edges $|\{(v_j, v_i) \in E\}|$ from any other node $v_j$
    - 'out': Out-degree $d_{out}(v_i)$ of node $v_i$, i.e. the number of outgoing edges $|\{(v_i, v_j) \in E\}|$ to any other node $v_j$
    - 'total': Total degree $d_{total}(v_i)$ of node $v_i$, i.e. the sum of in-degree and out-degree $d_{in}(v_i) + d_{out}(v_i)$

    The degree sequence is returned as a `torch.Tensor` of shape `(n,)` where $n$ is the number of nodes in the graph.
    The order of the degree sequence corresponds to the indexing of the nodes in the graph and
    the index of a node $v_i$ given by its label can be accessed via [`graph.mapping.to_idx(v_i)`][pathpyG.core.index_map.IndexMap.to_idx].

    ??? reference
        For further reading, see Chapter 10.3 in *Networks*[^1] by Mark Newman.

    [^1]: *Newman, M. E. J. Networks. (Oxford University Press, 2018). doi:10.1093/oso/9780198805090.001.0001.*

    Args:
        graph: The `Graph` object for which degrees are calculated
        mode:  'in', 'out' or 'total' for directed graphs, ignored for undirected graphs

    Returns:
        torch.Tensor: A tensor containing the degree sequence
    """
    if mode == "total":
        if graph.is_directed():
            return graph.degrees(mode="in", return_tensor=True) + graph.degrees(mode="out", return_tensor=True)  # type: ignore[operator]
        else:
            return graph.degrees(mode="in", return_tensor=True)  # type: ignore[return-value]
    else:
        return graph.degrees(mode, return_tensor=True)  # type: ignore[return-value]


def degree_distribution(graph: Graph, mode: str = "total") -> torch.Tensor:
    r"""Calculates the (unweighted) degree distribution of a [graph][pathpyG.core.graph.Graph].

    The degree distribution $P_{mode}(d)$ of a [graph][pathpyG.core.graph.Graph] $G=(V, E)$ is defined as

    $$
    P_{mode}(d) = \frac{N_d}{n}
    $$

    with $N_d = |\{v_i \in V : d_{mode}(v_i) = d\}|$ being the number of nodes with degree $d$ and $n$ being the total number of nodes in the graph.
    The modes are defined as follows:

    - 'in': In-degree $d_{in}(v_i)$ of node $v_i$, i.e. the number of incoming edges $|\{(v_j, v_i) \in E\}|$ from any other node $v_j$
    - 'out': Out-degree $d_{out}(v_i)$ of node $v_i$, i.e. the number of outgoing edges $|\{(v_i, v_j) \in E\}|$ to any other node $v_j$
    - 'total': Total degree $d_{total}(v_i)$ of node $v_i$, i.e. the sum of in-degree and out-degree $d_{in}(v_i) + d_{out}(v_i)$

    The degree distribution is returned as a `torch.Tensor` of shape `(d_max + 1,)` where `d_max` is the maximum degree in the graph.

    ??? reference
        For further reading, see Chapter 10.3 in *Networks*[^1] by Mark Newman.

    [^1]: *Newman, M. E. J. Networks. (Oxford University Press, 2018). doi:10.1093/oso/9780198805090.001.0001.*

    Args:
        graph: The `Graph` object for which the degree distribution is calculated
        mode:  'in', 'out' or 'total' for directed graphs, ignored for undirected graphs

    Returns:
        torch.Tensor: A tensor containing the degree distribution
    """
    return degree_sequence(graph, mode=mode).bincount() / graph.n


def degree_raw_moment(graph: Graph, k: int = 1, mode: str = "total") -> float:
    r"""Calculates the k-th raw moment of the degree distribution of a [graph][pathpyG.core.graph.Graph].

    The k-th raw moment $\langle d^k \rangle$ of the [degree distribution][pathpyG.statistics.degrees.degree_distribution] $P_{mode}(d)$ of a [graph][pathpyG.core.graph.Graph] $G=(V, E)$ is defined as

    $$
    \langle d_{mode}^k \rangle = \sum_d d^k P_{mode}(d).
    $$

    ??? reference
        For further reading, see Equation 10.20 in Chapter 10.4 in *Networks*[^1] by Mark Newman.

    [^1]: *Newman, M. E. J. Networks. (Oxford University Press, 2018). doi:10.1093/oso/9780198805090.001.0001.*

    Args:
        graph:  The graph in which to calculate the k-th raw moment
        k: The order of the moment to calculate
        mode:  'in', 'out' or 'total' for directed graphs, ignored for undirected graphs

    Returns:
        float: The k-th raw moment of the degree distribution
    """
    p_k = degree_distribution(graph, mode=mode)
    x = torch.arange(len(p_k), dtype=torch.float32)
    m = torch.sum((x**k) * p_k).item()
    return m


def mean_degree(graph: Graph, mode: str = "total") -> float:
    r"""Calculates the mean degree of a [graph][pathpyG.core.graph.Graph].

    The mean degree $\langle d \rangle$ of a [graph][pathpyG.core.graph.Graph] $G=(V, E)$ is defined as

    $$
    \langle d_{mode} \rangle = \frac{1}{n} \sum_{i=1}^{n} d_{mode}(v_i)
    $$

    with $d_{mode}(v_i)$ being the degree of node $v_i$ in mode 'in', 'out' or 'total'.
    The modes are defined as follows:

    - 'in': In-degree $d_{in}(v_i)$ of node $v_i$, i.e. the number of incoming edges $|\{(v_j, v_i) \in E\}|$ from any other node $v_j$
    - 'out': Out-degree $d_{out}(v_i)$ of node $v_i$, i.e. the number of outgoing edges $|\{(v_i, v_j) \in E\}|$ to any other node $v_j$
    - 'total': Total degree $d_{total}(v_i)$ of node $v_i$, i.e. the sum of in-degree and out-degree $d_{in}(v_i) + d_{out}(v_i)$

    Args:
        graph: The graph for which to calculate the mean degree
        mode:  'in', 'out' or 'total' for directed graphs, ignored for undirected graphs

    Returns:
        float: The mean degree of the graph
    """
    return torch.mean(degree_sequence(graph, mode=mode).float()).item()


def mean_neighbor_degree(graph: Graph, mode: str = "total", exclude_backlink=False) -> float:
    r"""Calculates the mean neighbor degree of a [graph][pathpyG.core.graph.Graph].

    The mean neighbor degree $\langle d_{\mathcal{N}} \rangle$ of a [graph][pathpyG.core.graph.Graph] $G=(V, E)$ is defined as

    $$
    \langle d_{\mathcal{N}} \rangle = \frac{1}{m} \sum_{v_i \in V} \sum_{v_j \in \mathcal{N}(v_i)} d_{mode}(v_j)
    $$

    with the number of edges [$m$][pathpyG.core.graph.Graph.m], the set of neighbors $\mathcal{N}(v_i)$ of node $v_i$ and $d_{mode}(v_j)$ being the degree of neighbor node $v_j \in \mathcal{N}(v_i)$ in mode 'in', 'out' or 'total'.
    The modes are defined as follows:

    - 'in': In-degree $d_{in}(v_i)$ of node $v_i$, i.e. the number of incoming edges $|\{(v_j, v_i) \in E\}|$ from any other node $v_j$
    - 'out': Out-degree $d_{out}(v_i)$ of node $v_i$, i.e. the number of outgoing edges $|\{(v_i, v_j) \in E\}|$ to any other node $v_j$
    - 'total': Total degree $d_{total}(v_i)$ of node $v_i$, i.e. the sum of in-degree and out-degree $d_{in}(v_i) + d_{out}(v_i)$

    ??? reference
        For further reading, see Chapter 12.2 in *Networks*[^1] by Mark Newman.

    [^1]: *Newman, M. E. J. Networks. (Oxford University Press, 2018). doi:10.1093/oso/9780198805090.001.0001.*

    Args:
        graph: The graph for which to calculate the mean neighbor degree
        mode:  'in', 'out' or 'total' for directed graphs, ignored for undirected graphs
        exclude_backlink: Whether to exclude the backlink to the original node when calculating neighbor degrees

    Returns:
        float: The mean neighbor degree of the graph
    """
    in_degree = degree_sequence(graph, mode="in")
    degree_seq = degree_sequence(graph, mode=mode)
    if exclude_backlink:
        degree_seq = degree_seq - 1
    return torch.sum(in_degree * degree_seq).item() / graph.m


def degree_central_moment(graph: Graph, k: int = 1, mode: str = "total") -> float:
    r"""Calculates the k-th central moment of the [degree distribution][pathpyG.statistics.degrees.degree_distribution].

    The k-th central moment $\mu_k$ of the [degree distribution][pathpyG.statistics.degrees.degree_distribution] $P_{mode}(d)$ of a [graph][pathpyG.core.graph.Graph] $G=(V, E)$ is defined as

    $$
    \mu_k = \sum_d (d - \langle d_{mode} \rangle)^k P_{mode}(d).
    $$

    where $\langle d_{mode} \rangle$ is the [mean degree][pathpyG.statistics.degrees.mean_degree] of the graph and $P_{mode}(d)$ is the [degree distribution][pathpyG.statistics.degrees.degree_distribution].

    Note:
        The 2nd central moment corresponds to the variance of the degree distribution.

    Args:
        graph: The graph for which to calculate the k-th central moment
        k: The order of the moment to calculate
        mode:  'in', 'out' or 'total' for directed graphs, ignored for undirected graphs

    Returns:
        float: The k-th central moment of the degree distribution
    """
    p_k = degree_distribution(graph, mode=mode)
    mean = mean_degree(graph, mode=mode)
    x = torch.arange(len(p_k), dtype=torch.float32)
    m = torch.sum((x - mean) ** k * p_k).item()
    return m


def degree_assortativity(graph: Graph, mode: str = "total") -> float:
    r"""Calculate the degree assortativity coefficient of the [graph][pathpyG.core.graph.Graph].

    The degree assortativity coefficient $r$ of a [graph][pathpyG.core.graph.Graph] $G=(V, E)$ is defined as

    $$
    r = \frac{\sum_{i,j} \left(A_{ij} - d_i d_j / (2m)\right) d_i d_j}{\sum_{i,j} \left(d_i \delta_{ij} - d_i d_j / (2m)\right) d_i d_j}
    $$

    with the adjacency matrix $A$, the degree $d_i$ of node $i$, the number of edges [$m$][pathpyG.core.graph.Graph.m] and the Kronecker delta $\delta_{ij}$.

    For computational reasons, we calculate the coefficient as follows:

    $$
    r = \frac{S_1S_e - S_2^2}{S_1S_3 - S_2^2}
    $$

    where $S_l = \sum_{i} d_i^l$ for $l=1,2,3$ and $S_e = \sum_{(i,j) \in E} d_i d_j$.

    ??? reference
        You can find the defintions above with further explanations in Equations (10.27) and (10.28) in Chapter 10.7 in *Networks*[^1] by Mark Newman.

    [^1]: *Newman, M. E. J. Networks. (Oxford University Press, 2018). doi:10.1093/oso/9780198805090.001.0001.*

    Args:
        graph: The graph for which to calculate the degree assortativity
        mode:  'in', 'out' or 'total' for directed graphs, ignored for undirected graphs

    Returns:
        float: The degree assortativity coefficient of the graph
    """
    degree_seq = degree_sequence(graph, mode=mode).float()
    S_1 = torch.sum(degree_seq).item()
    S_2 = torch.sum(degree_seq**2).item()
    S_3 = torch.sum(degree_seq**3).item()
    S_e = torch.sum(degree_seq[graph.data.edge_index[0]] * degree_seq[graph.data.edge_index[1]]).item()

    numerator = S_1 * S_e - S_2**2
    denominator = S_1 * S_3 - S_2**2

    return numerator / denominator


def degree_generating_function(
    graph: Graph, x: float | list[float] | _np.ndarray | torch.Tensor, mode: str = "total"
) -> float | torch.Tensor:
    r"""Returns the generating function of the degree distribution of a [graph][pathpyG.core.graph.Graph].

    Returns $f(x)$ where $f$ is the probability generating function for the degree
    distribution $P_{mode}(d)$ for a [graph][pathpyG.core.graph.Graph] $G=(V, E)$ defined as
    
    $$
    f(x) = \sum_d P_{mode}(d) x^d.
    $$

    The function is defined in the interval $\left[0,1\right]$. The following properties hold:

    1. The values of the [degree distribution][pathpyG.statistics.degrees.degree_distribution] $P_{mode}(d)$ can be retrieved from the generating function via
        $$
        P_{mode}(d) = \left[\frac{1}{d!} \frac{d^d f}{dx^d}\right]_{x=0}
        $$

        with $\frac{d^d}{dx^d} f$ being the d-th derivative of $f$ by $x$.

    2. The $k$-th raw moment of the degree distribution can be retrieved from the generating function with

        $$
        \left[\left(x \frac{d}{dx}\right)^k f\right]_{x=1} = \langle d_{mode}^k \rangle.
        $$

    Args:
        graph: The [graph][pathpyG.core.graph.Graph] for which the generating function shall be computed
        x:  float, list, numpy.ndarray, or torch.Tensor
            The argument(s) for which value(s) $f(x)$ shall be computed.
        mode:  'in', 'out' or 'total' for directed graphs, ignored for undirected graphs

    Returns:
        float or torch.Tensor: The value(s) of the generating function at x

    Examples:
        Generate simple network:

        >>> import pathpyG as pp
        >>>
        >>> g = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'd'),
        >>>                              ('d', 'e'), ('d', 'f'), ('e', 'f')]).to_undirected()

        Return single function value:

        >>> val = pp.statistics.degree_generating_function(g, 0.3)
        >>> print(val)
        0.069

        Plot generating function of degree distribution

        >>> x = list(range(10))
        >>> y = pp.statistics.degree_generating_function(g, x)
        >>> print(y)
        tensor([  0.0000,   1.0000,   5.3333,  15.0000,  32.0000,  58.3333,  96.0000,
        147.0000, 213.3333, 297.0000])
    ```
    """
    p_k = degree_distribution(graph, mode=mode)

    if isinstance(x, float):
        x_range = torch.tensor([x])
    elif isinstance(x, list) or isinstance(x, _np.ndarray):
        x_range = torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, torch.Tensor):
        x_range = x.float()
    else:
        raise TypeError("x must be a float, list, numpy.ndarray, or torch.Tensor")

    # Via broadcasting, compute f(x) for all x in x_range defined as f(x) = sum_k p_k * x^k
    values = torch.sum(p_k.unsqueeze(1) * (x_range.unsqueeze(0) ** torch.arange(p_k.size(0)).unsqueeze(1)), dim=0)

    if isinstance(x, float):
        return values[0].item()
    else:
        return values
