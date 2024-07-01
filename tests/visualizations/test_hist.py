from pathpyG.core.graph import Graph
from pathpyG.visualisations.hist_plots import hist


def test_hist_plot() -> None:
    """Test to plot a histogram."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])
    deg = net.degrees()

    # print(deg)
    # hist(net)
