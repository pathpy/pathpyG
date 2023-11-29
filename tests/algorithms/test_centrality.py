from pathpyG.core.Graph import Graph
from pathpyG.visualisations.hist_plots import hist
from pathpyG.algorithms import centrality

def test_centrality(simple_graph):
    r = centrality.closeness_centrality(simple_graph)
    assert r == {'a': 0, 'b': 0.5, 'c': 1}
