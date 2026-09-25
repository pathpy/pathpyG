import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # General Concepts of the Tensor-based Implementations

    ## Prerequisites

    First, we need to set up our Python environment that has PyTorch, PyTorch Geometric and PathpyG installed. Depending on where you are executing this notebook, this might already be (partially) done. E.g. Google Colab has PyTorch installed by default so we only need to install the remaining dependencies. The DevContainer that is part of our GitHub Repository on the other hand already has all of the necessary dependencies installed.

    In the following, we install the packages for usage in Google Colab using Jupyter magic commands. For other environments comment in or out the commands as necessary. For more details on how to install `pathpyG` especially if you want to install it with GPU-support, we refer to our [documentation](https://www.pathpy.net/dev/getting_started/). Note that `%%capture` discards the full output of the cell to not clutter this tutorial with unnecessary installation details. If you want to print the output, you can comment `%%capture` out.
    """)
    return


@app.cell
def _():
    # %%capture
    # # !pip install torch
    # # !pip install torch_geometric
    # # !pip install git+https://github.com/pathpy/pathpyG.git
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Motivation and Learning Objectives

    The inner workings of the core classes of PathpyG are based on tensor operations provided by PyTorch and PyTorch Geometric. Especially the creation of higher-order structures using the lift-order functions and the `MultiOderModel` heavily rely on tensor operations for efficiency reasons. While these implementations are highly optimized, they are very hard to read and understand for newcomers. This tutorial aims to explain the general concepts and ideas behind these implementations in a more accessible way. Additionally, we will provide step-by-step explanations of the core functions in the following sections.
    """)
    return


@app.cell
def _():
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import cumsum, degree, sort_edge_index

    import pathpyG as pp

    return Data, cumsum, degree, pp, sort_edge_index, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Order-lifting and Line Graph Transformations

    At the core of creating higher-order models is the `lift_order_edge_index` function that is essentially a line graph transformation. Given an edge index of a graph and the number of nodes in the graph, this function creates the edge index for the corresponding line graph. Let's look at an example:
    """)
    return


@app.cell
def _(pp, torch):
    mapping = pp.IndexMap(list("abcdef"))
    graph = pp.Graph.from_edge_index(
        edge_index=torch.tensor([[0, 1, 3, 4, 2, 2, 5], [2, 2, 5, 5, 3, 4, 0]]), mapping=mapping
    )
    pp.plot(graph);
    return graph, mapping


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can create the line graph for this graph using the `lift_order_edge_index` function as follows:
    """)
    return


@app.cell
def _(Data, graph, pp):
    _second_order_edge_index = pp.algorithms.lift_order.lift_order_edge_index(edge_index=graph.data.edge_index, num_nodes=graph.n)
    second_order_mapping = pp.IndexMap(graph.edges)
    second_order_data = Data(edge_index=_second_order_edge_index, node_sequence=graph.data.edge_index.t())
    line_graph = pp.Graph(data=second_order_data, mapping=second_order_mapping)
    pp.plot(line_graph)
    return (line_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To create the higher-order `PathpyG.Graph`, we needed to specify a `node_sequence` in the `Data` object. The node sequence above was given by the original edges of the graph. This `node_sequence` keeps track of which original nodes correspond to which higher-order nodes in the higher-order graph. In a second order graph, each higher-order node corresponds to an edge in the original graph. In a graph of order k, each higher-order node corresponds to a path of length k in the original graph. With this, we can always trace back which higher-order node corresponds to which original nodes.

    As long as we have this mapping from higher-order nodes to original nodes, we can always do an additional line graph transformation to create even higher order graphs. Below, we create a third-order graph:
    """)
    return


@app.cell
def _(Data, graph, line_graph, pp, torch):
    third_order_edge_index = pp.algorithms.lift_order.lift_order_edge_index(edge_index=line_graph.data.edge_index, num_nodes=line_graph.n)
    third_order_data = Data(edge_index=third_order_edge_index, node_sequence=torch.cat([line_graph.data.node_sequence[line_graph.data.edge_index[0]], line_graph.data.node_sequence[line_graph.data.edge_index[1]][:, -1:]], dim=1))
    third_order_mapping = pp.IndexMap([tuple(seq) for seq in graph.mapping.to_ids(third_order_data.node_sequence).tolist()])
    third_order_graph = pp.Graph(data=third_order_data, mapping=third_order_mapping)
    pp.plot(third_order_graph);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that above, we constructed the `node_sequence` for the third-order graph by concatenating the sequences of the two nodes that form each edge in the second-order graph. However, only the first node in the sequence of the higher-order source and the last node in the sequence of the higher-order target node are different. The middle nodes are the same for both higher-order nodes since they represent the overlapping part of the paths.

    ### Under the Hood of `lift_order_edge_index`

    Let us now take a closer look at how the `lift_order_edge_index` function works under the hood. The whole function essentially only needs 10 lines of code and looks as follows:
    """)
    return


@app.cell
def _(cumsum, degree, torch):
    def lift_order_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Lift order function as implemented in PathpyG."""
        outdegree = degree(edge_index[0], dtype=torch.long, num_nodes=num_nodes)
        outdegree_per_dst = outdegree[edge_index[1]]
        num_new_edges = outdegree_per_dst.sum()
        ho_edge_srcs = torch.repeat_interleave(outdegree_per_dst)
        ptrs = cumsum(outdegree, dim=0)[:-1]
        ho_edge_dsts = torch.repeat_interleave(ptrs[edge_index[1]], outdegree_per_dst)
        idx_correction = torch.arange(num_new_edges, dtype=torch.long)
        idx_correction = idx_correction - cumsum(outdegree_per_dst, dim=0)[ho_edge_srcs]
        ho_edge_dsts = ho_edge_dsts + idx_correction
        return torch.stack([ho_edge_srcs, ho_edge_dsts], dim=0)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    However, what the function does exactly is obfuscated by the heavy use of tensor operations. Let us break down the function step-by-step to understand what is happening internally.

    <div class="admonition note">
        <p class="admonition-title">Note</p>

            Due to the high complexity of the tensor operations, we will maintain to lines of explanations that try to explain the same concepts with different words. One explanation line will be added to the code snippets as comments and the other explanation line will be provided in the markdown cells between the code snippets.


    </div>

    <div class="admonition warning">
        <p class="admonition-title">Edge index must be sorted!</p>

        The <code>lift_order_edge_index</code> function assumes that the input <code>edge_index</code> is sorted by source nodes. This is not enforced by the function itself, because we ensure that the edge indices are sorted whenever we create a <code>PathpyG.Graph</code> object. This step is crucial for the correct functioning of the <code>lift_order_edge_index</code> function.


    </div>

    1. The function first computes the outdegree of each node in the graph using the `degree` function from `torch_geometric.utils`. This gives us a tensor containing the number of outgoing edges for each node.
    """)
    return


@app.cell
def _(degree, graph, torch):
    # Compute the outdegree of each node used to get all the edge combinations leading to a higher-order edge
    outdegree = degree(graph.data.edge_index[0], dtype=torch.long, num_nodes=graph.n)
    print("Outdegree per node:")
    for node in graph.nodes:
        print(f"\t{node}: {outdegree[graph.mapping.to_idx(node)].item()}")
    return (outdegree,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2. Next, we map the outdegree values to the destination nodes of each edge in the edge index. This gives us a tensor where each entry corresponds to the outdegree of the target node of each edge.

    <div class="admonition note">
        <p class="admonition-title">Note</p>

            This helps us because for the line graph transformation, we need to transform each edge into a node and then connect these nodes (previously edges) if a node in the original graph connects them. Therefore, we need to create a higher-order edge for each combination of incoming and outgoing edges for each node in the original graph. The outdegree of the target node tells us how many outgoing edges there are for each target node, which directly translates to how many higher-order edges we need to create for each incoming edge.


    </div>
    """)
    return


@app.cell
def _(graph, outdegree):
    # For each center node, we need to combine each outgoing edge with each incoming edge
    # We achieve this by creating `outdegree` number of edges for each destination node 
    # of the old edge index
    outdegree_per_dst = outdegree[graph.data.edge_index[1]]
    print("\nOutdegree per destination node of each edge:")
    for e, outdeg in zip(graph.edges, outdegree_per_dst.tolist()):
        print(f"\t{e}: {outdeg}")
    return (outdegree_per_dst,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3. Next, we create the source nodes for the higher-order graph. For this, we create a new index that maps the original edges to its index as a higher-order node. This is done by creating a range from 0 to the number of edges in the original graph. We then repeat each index according to the outdegree of the corresponding target node. This way, we create a source node for each combination of incoming and outgoing edges for each target node, which will be the edges in the higher-order graph.
    """)
    return


@app.cell
def _(graph, outdegree_per_dst, torch):
    # Use each edge from the edge index as node and assign the new indices in the order of the original edge index
    # Each higher order node has one outgoing edge for each outgoing edge of the original destination node
    # Since we keep the ordering, we can just repeat each node using the `outdegree_per_dst` tensor
    ho_edge_srcs = torch.repeat_interleave(outdegree_per_dst)
    print("\nHigher-order edge source indices:\n", ho_edge_srcs.tolist())
    print("Higher-order edge sources:\n", graph.mapping.to_ids(graph.data.edge_index[:, ho_edge_srcs]).T)
    return (ho_edge_srcs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4. Now, we need to create the target nodes for the higher-order edges. For this, we first need to know where the edges of each node start in the original edge index. We can compute this by calculating the cumulative sum of the outdegree values of all nodes. This gives us a tensor where each entry corresponds to the starting index of the edges for each node in the original edge index.

    <div class="admonition tip">
        <p class="admonition-title">Cumulative Sum</p>

            There is one <code>cumsum</code> implementation in PyTorch and one in PyTorch Geometric. The one in PyTorch Geometric starts with an initial zero value, while the one in PyTorch does not. This means that the <code>torch.cumsum</code> function will give us the end pointers of the edges for each node, while the <code>torch_geometric.utils.cumsum</code> function will give us the start pointers (including a last pointer that is equal to the total number of edges). Therefore, we use the <code>torch_geometric.utils.cumsum</code> function here and remove the last entry afterwards.


    </div>
    """)
    return


@app.cell
def _(cumsum, outdegree):
    # For each node, we calculate pointers of shape (num_nodes,) that indicate the start of the original edges
    # (new higher-order nodes) that have the node as source node
    ptrs = cumsum(outdegree, dim=0)[:-1]
    print("Edge start pointers per node:\n", ptrs.tolist())
    return (ptrs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    5. With the starting pointers of the edges for each node, we can start with the creation of the target nodes for the higher-order edges. Remember that we assigned the node indices based on the order of edges in the original edge index and ordered the higher-order source nodes accordingly. Therefore, we are essentially going through each edge, and combine it with each outgoing edge of the edges target node to create the higher-order edges. Since the edges are **ordered** by source nodes, we are going through all nodes in the original graph in order by going through each outgoing edge of each node. This means that for each edge in the original graph, we can look up where the outgoing edges of its target node start in the original edge index using the `ptrs` tensor we created in the previous step. We then repeat these starting pointers according to the outdegree of the corresponding target node to create a target node for each combination of incoming and outgoing edges for each target node.
    """)
    return


@app.cell
def _(graph, outdegree_per_dst, ptrs, torch):
    # Use these pointers to get the start of the edges for each higher-order src and repeat it `outdegree` times
    # Since we keep the ordering, all new higher-order edges that have the same src are indexed consecutively
    ho_edge_dsts = torch.repeat_interleave(ptrs[graph.data.edge_index[1]], outdegree_per_dst)
    print('Higher-order edge destination indices (before correction):\n', ho_edge_dsts.tolist())
    return (ho_edge_dsts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    6. For now, we do not have the correct indices for the higher-order target nodes yet. Since we only repeated the starting pointers of the edges for each target node, we only have the correct offsets for each group of higher-order edges corresponding to each target node. However, within each group, we need to assign the correct indices to the higher-order target nodes. Luckily, we only need to count up from the starting pointer for each group corresponding to one incoming edge in the original graph due to the ordering of the edges. For this, we create a correction index that counts up from 0 to the total number of higher-order edges.
    """)
    return


@app.cell
def _(ho_edge_srcs, torch):
    # Since the above only repeats the start of the edges, we need to add (0, 1, 2, 3, ...)
    # for all `outdegree` number of edges consecutively to get the correct destination nodes
    # We can achieve this by starting with a range from (0, 1, ..., num_new_edges)
    idx_correction = torch.arange(ho_edge_srcs.size(0), dtype=torch.long)
    print("Index correction (before adjustment):\n", idx_correction.tolist())
    return (idx_correction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    7. We then subtract the cumulative sum of the outdegree values of the higher-order source nodes from this correction index. This effectively resets the counting for each group of higher-order edges corresponding to each target node.
    """)
    return


@app.cell
def _(cumsum, ho_edge_srcs, idx_correction, outdegree_per_dst):
    # Then, we subtract the cumulative sum of the outdegree for each destination node
    idx_correction_1 = idx_correction - cumsum(outdegree_per_dst, dim=0)[ho_edge_srcs]
    print('Index correction (after adjustment):\n', idx_correction_1.tolist())
    return (idx_correction_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    8. Finally, we add this correction index to the starting pointers of the edges for each target node to get the correct indices for the higher-order target nodes.
    """)
    return


@app.cell
def _(graph, ho_edge_dsts, idx_correction_1):
    _ho_edge_dsts = ho_edge_dsts + idx_correction_1
    print('Higher-order edge destination indices (after correction):\n', _ho_edge_dsts.tolist())
    print('Higher-order edge destinations:\n', graph.mapping.to_ids(graph.data.edge_index[:, _ho_edge_dsts]).T)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This gives us the final higher-order edge index that we can return from the function.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Temporal Order Lifting

    One of the core functionalities of PathpyG is the ability to create temporal higher-order models. For this, an extension of the `lift_order_edge_index` function to temporal graphs is needed. We implement this in the `EventGraph.build_edge_index` function. This function works similarly to the `lift_order_edge_index` function, but with some additional steps to account for the temporal aspect of the graph. The main difference is that we need to ensure that the higher-order edges respect the temporal ordering of the original edges. Let us take a look at an example:
    """)
    return


@app.cell
def _(pp):
    tedges = [
        ("a", "b", 1),
        ("a", "b", 2),
        ("b", "a", 3),
        ("b", "c", 3),
        ("d", "c", 4),
        ("a", "b", 4),
        ("c", "b", 4),
        ("c", "d", 5),
        ("b", "a", 5),
        ("c", "b", 6),
    ]
    t = pp.TemporalGraph.from_edge_list(tedges)
    pp.plot(t);
    return (t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can create a second-order graph from this temporal graph using the `EventGraph.build_edge_index` function. This second-order graph is typically referred to as an event graph. Each node in the graph is an event (edge) in the original temporal graph and two events are connected if they can follow each other in time respecting a maximum time difference `delta`. Here, we set `delta=2` which means that two events can be connected if the time difference between them is at most 2 time units.
    """)
    return


@app.cell
def _(Data, graph, pp, t):
    event_edge_index = pp.core.event_graph.EventGraph.build_edge_index(t, delta=2)
    event_mapping = pp.IndexMap(t.temporal_edges)
    event_data = Data(edge_index=event_edge_index, node_sequence=graph.data.edge_index.t())
    event_graph = pp.Graph(data=event_data, mapping=event_mapping)
    pp.plot(event_graph);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Starting with the event graph, we have a static higher-order representation of the temporal graph that we can use to create higher-order models. For each following lift-order transformations, we can use the same principles as described in the previous section on order-lifting and line graph transformations.

    #### Internals of the `EventGraph.build_edge_index` Function

    The simplest way to implement the `EventGraph.build_edge_index` function would be to first create the full higher-order edge index using the `lift_order_edge_index` function and then filter out the edges that do not respect the temporal ordering. The filter function could look as follows:
    """)
    return


@app.cell
def _(torch):
    def filter_time_respecting_edges(event_edge_index: torch.Tensor, timestamps: torch.Tensor, delta: int) -> torch.Tensor:  # noqa: D103
        # Subtract timestamps of the two events to get the time difference
        time_diff = timestamps[event_edge_index[1]] - timestamps[event_edge_index[0]]
        # Create masks for filtering
        # Remove non-time-respecting higher-order edges
        non_negative_mask = time_diff > 0
        # Remove edges that are too far apart in time based on delta
        delta_mask = time_diff <= delta
        # Combine masks to get the final time-respecting edges
        time_respecting_mask = non_negative_mask & delta_mask
        # Filter the event_edge_index using the time_respecting_mask
        return event_edge_index[:, time_respecting_mask]

    return (filter_time_respecting_edges,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can combine the above filter function with the `lift_order_edge_index` function to create a lift-order function for temporal graphs as follows:

    <div class="admonition warning">
        <p class="admonition-title">Warning</p>

        If we use the standard <code>lift_order_edge_index</code> function, we need to ensure that the input edge index is sorted by source nodes because the <code>edge_index</code> of a <code>TemporalGraph</code> is sorted by time and not by source nodes.


    </div>
    """)
    return


@app.cell
def _(Data, filter_time_respecting_edges, pp, sort_edge_index, t):
    # Sort by source node indices
    sorted_edge_index, time = sort_edge_index(t.data.edge_index.as_tensor(), t.data.time)
    # Lift the edge index to the second order
    _second_order_edge_index = pp.algorithms.lift_order.lift_order_edge_index(edge_index=sorted_edge_index, num_nodes=t.n)
    # Filter the edges based on the lifted edge index
    filtered_edge_index = filter_time_respecting_edges(_second_order_edge_index, timestamps=time, delta=2)
    # Create `pp.Graph` from the filtered edge index
    filtered_event_mapping = pp.IndexMap([tuple([*t.mapping.to_ids(edge).tolist(), timestamp.item()]) for edge, timestamp in zip(sorted_edge_index.t(), time)])
    filtered_event_data = Data(edge_index=filtered_edge_index, node_sequence=sorted_edge_index.t())
    filtered_event_graph = pp.Graph(data=filtered_event_data, mapping=filtered_event_mapping)
    pp.plot(filtered_event_graph)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div class="admonition note">
        <p class="admonition-title">Note</p>

            The indexing of the above implementation is different from the one currently implemented in PathpyG. So while the illustrations look identical, the actual indices of the higher-order nodes will differ.


    </div>

    However, the above implementation has a large memory consumption for graphs with many edges because the full higher-order edge index is created before filtering. Therefore, we implement a more memory-efficient version in PathpyG that constructs the higher-order edges from the temporal graph sequentially for each timestamp. This implementation (available as `EventGraph.build_edge_index`) looks as follows:
    """)
    return


@app.cell
def _(pp, torch):
    def build_edge_index(g: pp.TemporalGraph, delta: int = 1):  # noqa: D103
        # first-order edge index
        edge_index, timestamps = g.data.edge_index, g.data.time

        delta = torch.tensor(delta, device=edge_index.device)  # type: ignore[assignment]
        indices = torch.arange(0, edge_index.size(1), device=edge_index.device)

        unique_t = torch.unique(timestamps, sorted=True)
        second_order = []

        # lift order: find possible continuations for edges in each time stamp
        for t in unique_t:
            # find indices of all source edges that occur at unique timestamp t
            src_time_mask = timestamps == t
            src_edge_idx = indices[src_time_mask]

            # find indices of all edges that can possibly continue edges occurring at time t for the given delta
            dst_time_mask = (timestamps > t) & (timestamps <= t + delta)
            dst_edge_idx = indices[dst_time_mask]

            if dst_edge_idx.size(0) > 0 and src_edge_idx.size(0) > 0:
                # compute second-order edges between src and dst idx
                # for all edges where dst in src_edges (edge_index[1, x[:, 0]]) matches src in dst_edges (edge_index[0, x[:, 1]])
                x = torch.cartesian_prod(src_edge_idx, dst_edge_idx)
                ho_edge_index = x[edge_index[1, x[:, 0]] == edge_index[0, x[:, 1]]]
                second_order.append(ho_edge_index)

        ho_index = torch.cat(second_order, dim=0).t().contiguous()
        return ho_index

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that above we do not use the same indexing trick that is used in the standard `lift_order_edge_index` function. Instead, we create all possible combinations of incoming and outgoing edges for all incoming edges at each timestamp. Therefore, we need a filtering step afterwards to ensure that only valid higher-order edges are created. However, we can skip the sorting step beforehand because we create all possible edge combinations using the cartesian product.

    It is also possible to combine both approaches, i.e., we create the higher-order edges for each timestamp separately using the indexing trick from the standard `lift_order_edge_index` function. While it saves the filtering step, it again requires sorting the edges beforehand which has been shown to be similar in performance to the above method. The code would look as follows:
    """)
    return


@app.cell
def _(cumsum, degree, pp, torch):
    def lift_order_temporal_combined(g: pp.TemporalGraph, delta: int=1):  # noqa: D103
        indices = torch.arange(0, g.data.edge_index.size(1))
        unique_t = torch.unique(g.data.time)
        second_order = []
        for i in range(unique_t.size(0)):
            t = unique_t[i]
            src_time_mask = g.data.time == t  # lift order: find possible continuations for edges in each time stamp
            src_edge_idx = indices[src_time_mask]
            dst_time_mask = (g.data.time > t) & (g.data.time <= t + delta)
            dst_node_mask = torch.isin(g.data.edge_index[0], g.data.edge_index[1, src_edge_idx])
            dst_edge_idx = indices[dst_time_mask & dst_node_mask]  # find indices of all source edges that occur at unique timestamp t
            if dst_edge_idx.size(0) > 0 and src_edge_idx.size(0) > 0:
                src_edges = g.data.edge_index[:, src_edge_idx]
                dst_edges = g.data.edge_index[:, dst_edge_idx]
                sorted_idx = torch.argsort(dst_edges[0])  # find indices of all edges that can possibly continue edges occurring at time t for the given delta
                dst_edge_idx = dst_edge_idx[sorted_idx]
                dst_edges = dst_edges[:, sorted_idx]
                outdegree = degree(dst_edges[0], dtype=torch.long, num_nodes=g.n)
                outdegree_per_dst = outdegree[src_edges[1]]
                num_new_edges = outdegree_per_dst.sum()
                ho_edge_srcs = torch.repeat_interleave(outdegree_per_dst)  # get sorted dst edges for efficient processing
                ptrs = cumsum(outdegree, dim=0)[:-1]
                ho_edge_dsts = torch.repeat_interleave(ptrs[src_edges[1]], outdegree_per_dst)
                idx_correction = torch.arange(num_new_edges, dtype=torch.long)
                idx_correction = idx_correction - cumsum(outdegree_per_dst, dim=0)[ho_edge_srcs]
                ho_edge_dsts = ho_edge_dsts + idx_correction
                second_order.append(torch.stack([src_edge_idx[ho_edge_srcs], dst_edge_idx[ho_edge_dsts]], dim=0))
        ho_index = torch.cat(second_order, dim=1)  # Use indexing trick to create higher-order edges
        return ho_index

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In contrast to the `lift_order_edge_index` implementation, the temporal version splits the edges into source and destination edges based on timestamps. For each timestamp, we select the edges that occur at that timestamp as source edges and all edges that occur at later timestamps (within the delta time window) as destination edges. Then, instead of repeating the higher-order source nodes for all edges, we only repeat them for the destination edges.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paths in PathpyG

    One other core functionality of PathpyG is the ability to work with paths. Paths are sequences of nodes that represent a walk through the graph. We show an example below:
    """)
    return


@app.cell
def _(cumsum, pp, torch):
    path_mapping = pp.IndexMap(list('abcde'))
    paths = pp.PathData(mapping=path_mapping)
    paths.append_walk(list('ab'))
    paths.append_walk(list('abd'))
    paths.append_walk(list('abec'))
    paths.append_walk(list('dbecb'))
    # To create an index map for path nodes, we need a unique identifier for each node in each path
    # There are two problems with this:
    # 1) The same node can appear in multiple paths (we solve this by using the path id)
    # 2) The same node can appear multiple times in the same path (we solve this by using the step id)
    _path_id = torch.repeat_interleave(paths.data.dag_num_nodes)
    _path_step_id = torch.arange(paths.data.node_sequence.size(0)) - cumsum(paths.data.dag_num_nodes)[:-1].repeat_interleave(paths.data.dag_num_nodes)
    _node_id = paths.mapping.to_ids(paths.data.node_sequence.flatten())
    path_node_mapping = pp.IndexMap([f'path {p}, step {s}, node {n}' for p, s, n in zip(_path_id.tolist(), _path_step_id.tolist(), _node_id.tolist())])
    pp.plot(pp.Graph.from_edge_index(paths.data.edge_index, mapping=path_node_mapping))
    return (paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `pp.PathData` is the core class for working with paths in PathpyG. It allows us to gather a collection of paths that are all walks on the same underlying graph. All paths are stored using one `edge_index` internally. Thus, two nodes in a path that both correspond to the same node in the underlying graph will **not** share the same index in the path graph. Instead, each occurrence of a node in a path is represented by a separate node in the path graph. This allows us to represent paths that visit the same node multiple times without ambiguity. The information about the underlying graph is stored in the internal `PathData.data.node_sequence` tensor, similar to higher-order graphs. Let us look at the example above to illustrate this:
    """)
    return


@app.cell
def _(paths):
    print("The paths represented using edge index look as follows:")
    for edge in paths.data.edge_index.t():
        print(
            f"\tInternal {edge.tolist()}: Underlying graph edge {paths.mapping.to_ids(paths.data.node_sequence[edge].view(-1)).tolist()}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `PathData` additionally stores some metadata about the paths so that you can easily access information about which nodes belong to which path. This includes

    - `dag_weight`: A tensor that stores the weight of each path (i.e., the number of times the path was observed).
    - `dag_num_edges`: A tensor that stores the number of edges in each path.
    - `dag_num_nodes`: A tensor that stores the number of nodes in each path.

    Using this information, you can, e.g., access the second path in the collection as follows:
    """)
    return


@app.cell
def _(paths):
    start = paths.data.dag_num_nodes[:1].sum().item()
    end = start + paths.data.dag_num_nodes[1].item()
    paths.mapping.to_ids(paths.data.node_sequence[start:end].view(-1)).tolist()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lastly, since we are using an `edge_index` internally, the `lift_order_edge_index` function works out-of-the-box for paths. A second-order representation of the paths can be created as follows:
    """)
    return


@app.cell
def _(paths):
    paths.mapping.to_ids(paths.data.node_sequence[paths.data.edge_index.t()].squeeze()).tolist()
    return


@app.cell
def _(cumsum, paths, pp, torch):
    _second_order_edge_index = pp.algorithms.lift_order.lift_order_edge_index(edge_index=paths.data.edge_index, num_nodes=paths.data.num_nodes)
    _path_id = torch.repeat_interleave(paths.data.dag_num_edges)
    _path_step_id = torch.arange(paths.data.dag_num_edges.sum()) - cumsum(paths.data.dag_num_edges)[:-1].repeat_interleave(paths.data.dag_num_edges)
    # Again, the index map would not be unique normally
    # Hence, we use the path id and step id (here a step is an edge in the path) to create a unique mapping for each higher-order node
    _node_id = paths.mapping.to_ids(paths.data.node_sequence[paths.data.edge_index.t()].squeeze())
    second_order_path_node_mapping = pp.IndexMap([f'path {p}, step {s}, edge {n}' for p, s, n in zip(_path_id.tolist(), _path_step_id.tolist(), _node_id.tolist())])
    second_order_paths = pp.Graph.from_edge_index(edge_index=_second_order_edge_index, mapping=second_order_path_node_mapping)
    pp.plot(second_order_paths)
    return (second_order_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-Order Models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the concepts above, we can now create multi-order models using the `MultiOrderModel` class. This class allows us to create higher-order models of arbitrary order from a given base temporal graph or paths. Let's look at an example of creating a multi-order model from a temporal graph:
    """)
    return


@app.cell
def _(pp, t):
    m_t = pp.MultiOrderModel.from_temporal_graph(t, max_order=2)
    pp.plot(m_t.layers[2]);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see that the second-order graph created by the `MultiOrderModel` is different from the one created by the `EventGraph.build_edge_index` function directly. This is because the `MultiOrderModel` higher-order DeBruijn graph representation. This representation merges higher-order nodes that correspond to the same path in the original graph. This means that temporal edges that appear in the event graph as different nodes will be merged into one node in the DeBruijn graph if they correspond to the same path in the original graph. This results in a more compact representation of the higher-order graph.

    The same is true for paths. We can create a multi-order model from a collection of paths as follows:
    """)
    return


@app.cell
def _(paths, pp):
    m_p = pp.MultiOrderModel.from_path_data(paths, max_order=2)
    pp.plot(m_p.layers[2]);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see that the higher-order node `a->b` which appeared thrice in the second-order graph created by the `lift_order_edge_index` function is now merged into one node in the DeBruijn graph representation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Internals of the `MultiOrderModel` Class

    Let us now take a closer look at how the `MultiOrderModel` class works under the hood. We already saw that the `MultiOrderModel` merges higher-order nodes from the line/event graph transformations.

    This is done in 3 distinct steps which we will go through using the paths example above:
    1. **Order Lifting**: First, we create the higher-order edge index using the appropriate lift-order function (`lift_order_edge_index` or `EventGraph.build_edge_index`) depending on whether we are working with paths or temporal graphs in the first order and `lift_order_edge_index` for the second order and beyond regardless of the input type.

    <div class="admonition note">
        <p class="admonition-title">Note</p>

            While we merge the higher-order nodes and aggregate the higher-order edges for each order, we need to use the original higher-order edge index to create the next order. This is because the transitivity of paths is only preserved in the original higher-order edge index.


    </div>
    """)
    return


@app.cell
def _(pp, second_order_paths):
    # We create the third-order representation of the paths
    third_order_edge_index_1 = pp.algorithms.lift_order.lift_order_edge_index(edge_index=second_order_paths.data.edge_index, num_nodes=second_order_paths.n)
    return (third_order_edge_index_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2. **Update Node Sequences**: Next, we need to update the internal `node_sequence` tensor to reflect the new higher-order nodes. For this, we create a new `node_sequence` by concatenating the last node of the target node sequence to the source node sequence. This way, we create a new sequence that corresponds to the paths represented by the next order nodes.
    """)
    return


@app.cell
def _(paths, second_order_paths, torch):
    second_order_node_sequence = paths.data.node_sequence[paths.data.edge_index.t()].squeeze()
    third_order_node_sequence = torch.cat([
        second_order_node_sequence[second_order_paths.data.edge_index[0]],
        second_order_node_sequence[second_order_paths.data.edge_index[1]][:, -1:]
    ], dim=1)
    return (third_order_node_sequence,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3. **Merge Higher-Order Nodes**: Finally, we need to merge the higher-order nodes that correspond to the same path in the original graph. For this, we create a unique mapping from the new `node_sequence` to unique indices. We can then use this mapping to update the higher-order edge index to reflect the merged nodes and then aggregate duplicate edges.
    """)
    return


@app.cell
def _(pp, third_order_edge_index_1, third_order_node_sequence):
    third_order_paths = pp.algorithms.lift_order.aggregate_edge_index(edge_index=third_order_edge_index_1, node_sequence=third_order_node_sequence)
    return (third_order_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After performing these steps, we can again visualize the resulting higher-order graph:
    """)
    return


@app.cell
def _(mapping, pp, third_order_paths):
    third_order_paths.mapping = pp.IndexMap([tuple(mapping.to_ids(v).tolist()) for v in third_order_paths.data.node_sequence])
    pp.plot(third_order_paths);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These steps can be repeated for each order until we reach the desired maximum order for the `MultiOrderModel`.

    ## Other Tensor-based Implementations

    The concepts from above can also be useful to implement other functionalities using tensor operations.

    ### Longest Path Extraction

    One example is the extraction of all longest paths from a directed acyclic graph (DAG). This can be done by iterating through all nodes in the DAG in topological order at the same time. We provide an example implementation below:
    """)
    return


@app.cell
def _(cumsum, degree, pp, torch):
    def get_all_paths_DAG(g: pp.Graph) -> dict:
        """Calculate all existing paths from any root node to any leaf node in a directed acyclic graph (DAG)."""
        paths_of_length = {}
        edge_index = g.data.edge_index.as_tensor()
        out_degree = degree(edge_index[0], num_nodes=g.n, dtype=torch.long)
        in_degree = degree(edge_index[1], num_nodes=g.n, dtype=torch.long)  # calculate degrees
        roots = torch.where(in_degree == 0)[0]
        leafs = out_degree == 0
        paths = roots.unsqueeze(1)
        paths_of_length[1] = paths[leafs[roots]].cpu().tolist()  # identify root nodes with in-degree zero
        paths = paths[~leafs[roots]]
        nodes = roots[~leafs[roots]]
        ptrs = cumsum(out_degree, dim=0)
        step = 1  # create path tensor that contains all paths that are not yet at a leaf node
        while nodes.size(0) > 0 or step > g.n:
            idx_repeat = torch.repeat_interleave(out_degree[nodes])  # remove all paths that are already at a leaf node
            next_idx = torch.repeat_interleave(ptrs[nodes], out_degree[nodes])
            idx_correction = torch.arange(next_idx.size(0), device=edge_index.device) - cumsum(out_degree[nodes], dim=0)[idx_repeat]  # continue all paths that are not at a leaf node
            next_idx = next_idx + idx_correction
            next_nodes = edge_index[1][next_idx]  # remember nodes that haven't been traversed yet
            paths = torch.cat([paths[idx_repeat], next_nodes.unsqueeze(1)], dim=1)
            paths_of_length[step] = paths[leafs[next_nodes]].tolist()
            paths = paths[~leafs[next_nodes]]
            nodes = next_nodes[~leafs[next_nodes]]
            step = step + 1  # count all longest paths in DAG
        return paths_of_length

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The function above starts at all root nodes (nodes with no incoming edges) and iteratively traverses all possible next nodes while keeping track of all current paths. Whenever a path reaches a leaf node (a node with no outgoing edges), it is added to the list of longest paths and removed from the current paths. This continues until all paths have reached a leaf node.

    <div class="admonition tip">
        <p class="admonition-title">Tip</p>

        Getting the next nodes for all current paths is done using a similar indexing trick as in the <code>lift_order_edge_index</code> function. This allows us to efficiently get all next nodes for all current paths in one go using tensor operations.


    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
