"""Iterator interface for rolling time window analysis in temporal graphs."""


class RollingTimeWindow:
    """An iterable rolling time window that can be used to perform time slice analysis of temporal graphs."""

    def __init__(self, temporal_graph, window_size, step_size=1, return_window=False, weighted=True):
        """Initialize RollingTimeWindow.
        
        Initialize a RollingTimeWindow instance that can be used to
        iterate through a sequence of time-slice networks for a given
        TemporalNetwork instance.

        Args:
            temporal_graph: TemporalGraphinstance that will be used to generate the
                sequence of time-slice networks.
            window_size: The width of the rolling time window used to create time-slice networks.
            step_size: The step size in time units by which the starting
                time of the rolling window will be incremented on each iteration.
            return_window: Whether or not the iterator shall return the current time window as a second return value. Default is False.
            weighted: Whether or not to return a weighted graph

        Example:
            ```py
            tedges = [('a', 'b', 1), ('b', 'c', 5), ('c', 'd', 9), ('c', 'e', 9),
              ('c', 'f', 11), ('f', 'a', 13), ('a', 'g', 18), ('b', 'f', 21),
              ('a', 'g', 26), ('c', 'f', 27), ('h', 'f', 27), ('g', 'h', 28),
              ('a', 'c', 30), ('a', 'b', 31), ('c', 'h', 32), ('f', 'h', 33),
              ('b', 'i', 42), ('i', 'b', 42), ('c', 'i', 47), ('h', 'i', 50)]
            t = pp.TemporalGraph.from_edge_list(tedges)
            r = pp.algorithms.RollingTimeWindow(t, 10, 10, return_window=True)
            for g, w in r:
                print('Time window ', w)
                print(g)
                print(g.data.edge_index)
                print('---')
            ```
        """
        self.g = temporal_graph
        self.window_size = window_size
        self.step_size = step_size
        self.current_time = self.g.start_time
        self.return_window = return_window
        self.weighted = weighted

    def __iter__(self):
        """Return the iterator object itself."""
        return self

    def __next__(self):
        """Return the next time-slice network in the rolling time window sequence."""
        if self.current_time <= self.g.end_time:
            time_window = (self.current_time, self.current_time + self.window_size)
            s = self.g.to_static_graph(weighted=self.weighted, time_window=time_window)
            self.current_time += self.step_size
            if self.return_window:
                return s, time_window
            else:
                return s
        else:
            raise StopIteration()
