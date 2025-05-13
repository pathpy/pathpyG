""" for testing purpose """

from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.visualisations import plot

tedges = [('a', 'b', 1),('a', 'b', 2), ('b', 'a', 3), ('b', 'c', 3), ('d', 'c', 4), ('a', 'b', 4), ('c', 'b', 4)]
t = TemporalGraph.from_edge_list(tedges)

plot(t, backend='manim')
