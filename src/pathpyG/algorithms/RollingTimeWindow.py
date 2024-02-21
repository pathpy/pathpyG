"""Iterator interface for rolling time window analysis in temporal graphs."""


from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Union, List
from collections import defaultdict

import numpy as np
import torch

from pathpyG import Graph
from pathpyG import TemporalGraph
from pathpyG import config


class RollingTimeWindow:
    r"""
    An iterable rolling time window that can be used to perform time slice
    analysis of temporal graphs.
    """

    def __init__(self, temporal_graph, window_size, step_size=1, return_window=False, weighted=True):
        r"""
        Initialises a RollingTimeWindow instance that can be used to
        iterate through a sequence of time-slice networks for a given
        TemporalNetwork instance.

        Parameters:
        -----------
        temporal_net:   TemporalNetwork
            TemporalNetwork instance that will be used to generate the
            sequence of time-slice networks.
        window_size:    int
            The width of the rolling time window used to create
            time-slice networks.
        step_size:      int
            The step size in time units by which the starting time of the rolling
            window will be incremented on each iteration. Default is 1.
        return_window: bool
            Whether or not the iterator shall return the current time window
            as a second return value. Default is False.

        Returns
        -------
        RollingTimeWindow
            An iterable sequence of tuples Network, [window_start, window_end]
        """
        self.g = temporal_graph
        self.window_size = window_size
        self.step_size = step_size
        self.current_time = self.g.start_time
        self.return_window = return_window
        self.weighted = weighted

    def __iter__(self):
        return self


    def __next__(self):
        if self.current_time <= self.g.end_time:
            time_window = (self.current_time, self.current_time+self.window_size)
            s = self.g.to_static_graph(weighted=self.weighted, time_window=time_window)
            self.current_time += self.step_size
            if self.return_window:
                return s, time_window
            else:
                return s
        else:
            raise StopIteration()
