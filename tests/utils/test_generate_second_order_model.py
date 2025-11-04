import torch

from pathpyG.core.multi_order_model import MultiOrderModel
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.utils.dbgnn import generate_second_order_model

def test_generate_second_order_model():
    tedges = [('a', 'b', 1), ('c', 'b', 1), ('c', 'a', 1), ('c', 'a', 1), ('f', 'c', 1), 
              ('b', 'c', 5), ('a', 'd', 5), ('c', 'd', 9), ('a', 'd', 9), ('c', 'e', 9),
              ('c', 'f', 11), ('f', 'a', 13), ('a', 'g', 18), ('b', 'f', 21),
              ('a', 'g', 26), ('c', 'f', 27), ('h', 'f', 27), ('g', 'h', 28),
              ('a', 'c', 30), ('a', 'b', 31), ('c', 'h', 32), ('f', 'h', 33),
              ('b', 'i', 42), ('i', 'b', 42), ('c', 'i', 47), ('h', 'i', 50)]
    
    g = TemporalGraph.from_edge_list(tedges)
    reference = MultiOrderModel.from_temporal_graph(g, max_order=2, delta=10).to_dbgnn_data() 
    
    g = TemporalGraph.from_edge_list(tedges)
    result = generate_second_order_model(g, delta=10).to_dbgnn_data()

    assert result.num_nodes == reference.num_nodes
    assert result.num_ho_nodes == reference.num_ho_nodes
    assert torch.equal(result.x, reference.x)
    assert torch.equal(result.edge_index, reference.edge_index)
    assert torch.equal(result.edge_weights, reference.edge_weights)
    assert torch.equal(result.x_h, reference.x_h)
    assert torch.equal(result.edge_index_higher_order, reference.edge_index_higher_order)
    assert torch.equal(result.edge_weights_higher_order, reference.edge_weights_higher_order)
    assert torch.equal(result.bipartite_edge_index, reference.bipartite_edge_index)