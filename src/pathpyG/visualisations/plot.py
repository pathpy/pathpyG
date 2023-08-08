"""Class to plot pathpy networks."""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : plot.py -- Module to plot pathpy networks
# Author    : Jürgen Hackl <hackl@ifi.uzh.ch>
# Time-stamp: <Thu 2021-08-26 14:15 juergen>
#
# Copyright (c) 2016-2019 Pathpy Developers
# =============================================================================
from __future__ import annotations
from typing import Any, List, Optional, Union, Dict, Callable
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import numpy as np

from pathpyG import config
from pathpyG.visualisations.utils import UnitConverter

from pathpyG.visualisations.backends import (D3js,
                                            Tikz,
                                            Matplotlib)

from pathpyG.visualisations.fileformats import (HTML,
                                                TEX,
                                                PDF,
                                                PNG)

from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph

# create logger for the Plot class
TIMESTAMP = config['temporal']['timestamp']

# config: defaultdict = defaultdict(dict)
# config['environment']['interactive'] = False

# General config
config['plot']['width'] = 800
config['plot']['height'] = 550
config['plot']['unit'] = 'px'
config['plot']['dpi'] = 96
config['plot']['margin'] = None
config['plot']['layout'] = 'force'
config['plot']['temporal'] = False
config['plot']['coordinates'] = False
config['plot']['euclidean'] = False
config['plot']['min_max_node_size'] = None
config['plot']['min_max_edge_size'] = None
config['plot']['keep_aspect_ratio'] = True

config['plot']['forceCharge'] = -20  # -30
config['plot']['forceRepel'] = -300  # -100
config['plot']['forceAlpha'] = 0.1
config['plot']['restartAlpha'] = 1
config['plot']['alphaMin'] = 0.001  # 0.1
config['plot']['targetAlpha'] = 0  # 0.2
config['plot']['chargeDistance'] = config['plot']['width']
config['plot']['repelDistance'] = 200
config['plot']['velocityDecay'] = 0.4  # .2
config['plot']['lookoutStrokeWidth'] = 1
config['plot']['lookoutOpacity'] = .5
config['plot']['lookoutWeight'] = 0.
config['plot']['radiusMinSize'] = 4
config['plot']['radiusMaxSize'] = 16
config['plot']['nodeTransitionDuration'] = 100
config['plot']['nodeTransitionDuration'] = 100
config['plot']['defaultEdgeWeight'] = 1

config['plot']['targetAlphaDragStarted'] = 0.3
config['plot']['targetAlphaDragEnd'] = 0.0


config['plot']['linkStrengthMin'] = 0.0
config['plot']['linkStrengthMax'] = .45


config['plot']['template'] = None
config['plot']['css'] = None

config['plot']['backend'] = ['tikz']
config['plot']['fileformat'] = ['tex']
config['plot']['latex_class_options'] = ''

config['plot']['interactiv'] = {}
config['plot']['interactiv']['backend'] = ['d3js']
config['plot']['interactiv']['fileformat'] = ['html']

# Animation config
config['plot']['animation'] = {}
config['plot']['animation']["enabled"] = False
config['plot']['animation']["start"] = None
config['plot']['animation']["end"] = None
config['plot']['animation']["steps"] = 20
config['plot']['animation']["speed"] = 100
config['plot']['animation']["unit"] = "seconds"

# Label config
config['plot']['label'] = {}
config['plot']['label']['centered'] = True
config['plot']['label']['enabled'] = True
config['plot']['label']['color'] = 'white'

config['plot']['label_centered'] = True
config['plot']['label_enabled'] = True
config['plot']['label_color'] = 'white'


# Node config
config['plot']['node'] = {}
config['plot']['node']['node_size'] = 15
config['plot']['node']['node_color'] = 'CornflowerBlue'
config['plot']['node']['node_opacity'] = .2
config['plot']['node']['id_as_label'] = True
config['plot']['node']['node_pos'] = None

config['plot']['curved'] = False
config['plot']['directed'] = False

# Edge config
config['plot']['edge'] = {}
config['plot']['edge']['edge_size'] = 2
config['plot']['edge']['edge_color'] = 'black'
config['plot']['edge']['edge_opacity'] = 1
config['plot']['edge']['directed'] = False
config['plot']['edge']['edge_curved'] = .5

# Widges config
config['plot']['widgets'] = {}

# tooltip
config['plot']['widgets']['tooltip'] = {}
config['plot']['widgets']['tooltip']['enabled'] = False
config['plot']['widgets']['tooltip']['size'] = '100px'

# save
config['plot']['widgets']['save'] = {}
config['plot']['widgets']['save']['title'] = 'Save'
config['plot']['widgets']['save']['enabled'] = True
config['plot']['widgets']['save']['tooltip'] = "Save the network as [svg] or [png]."

# zoom
config['plot']['widgets']['zoom'] = {}
config['plot']['widgets']['zoom']['title'] = 'Zoom'
config['plot']['widgets']['zoom']['enabled'] = True
config['plot']['widgets']['zoom']['tooltip'] = "Zoom-in with [+] <br> zoom-out with [-] <br> or reset zoom with [Reset]. <br> Furthermore, with [Shift+mouse wheel] you can also zoom."

# filter
config['plot']['widgets']['filter'] = {}
config['plot']['widgets']['filter']['title'] = 'Filter'
config['plot']['widgets']['filter']['enabled'] = False
config['plot']['widgets']['filter']['tooltip'] = "Filter the nodes base on given groups."
config['plot']['widgets']['filter']['groups'] = ["all"]

# search
config['plot']['widgets']['search'] = {}
config['plot']['widgets']['search']['title'] = 'Search'
config['plot']['widgets']['search']['enabled'] = True
config['plot']['widgets']['search']['tooltip'] = "Search for a node in the network."

# layout
config['plot']['widgets']['layout'] = {}
config['plot']['widgets']['layout']['title'] = 'Layout'
config['plot']['widgets']['layout']['enabled'] = False
config['plot']['widgets']['layout']['tooltip'] = "Change the layout of the Network. Per default a [Force] directed layout is used. If x and y coordinates are given, an [Coord] layout can be used."

# animation
config['plot']['widgets']['animation'] = {}
config['plot']['widgets']['animation']['title'] = 'Animation'
config['plot']['widgets']['animation']['enabled'] = True
config['plot']['widgets']['animation']['tooltip'] = "Play and pause animation of the temproal network."

# aggregation
config['plot']['widgets']['aggregation'] = {}
config['plot']['widgets']['aggregation']['title'] = 'Aggregation'
config['plot']['widgets']['aggregation']['enabled'] = True
config['plot']['widgets']['aggregation']['tooltip'] = "Aggregate time steps."
config['plot']['widgets']['aggregation']['past'] = 2
config['plot']['widgets']['aggregation']['future'] = 2
config['plot']['widgets']['aggregation']['aggregation'] = 1


def plot(obj, filename: Optional[str] = None,
         backend: Optional[str] = None, **kwargs) -> None:
    """Plot the object"""
    # initialize variables
    figure: Any

    # supported backends
    backends: Dict[str, Callable] = {
        'd3js': D3js,
        'tikz': Tikz,
        'matplotlib': Matplotlib
    }

    # supported file fileformats and corresponding default backends
    figures: Dict[str, Dict[str, Callable]] = {
        'html': {'fileformat': HTML, 'backend': D3js},
        'tex': {'fileformat': TEX, 'backend': Tikz},
        # 'csv': {'fileformat': CSV, 'backend': Tikz},
        'pdf': {'fileformat': PDF, 'backend': Tikz},
        'png': {'fileformat': PNG, 'backend': Matplotlib},
    }

    # initialize object parser
    parser: Parser = Parser()

    # check object
    try:
        if obj.N == 0:
            print('An empty network cannot be plotted. Please add at least one Node object.')
            return
    except Exception:
        print('The provided object cannot be plotted.')
        raise NotImplementedError

    _config = deepcopy(config['plot'])
    # _config = config['plot']
    # parse object to json like dict
    data: defaultdict = parser(obj, _config, **kwargs)

    # check filename
    # if no file name is given
    if filename is None:
        # generate default html figure with d3js
        figure = HTML()
        figure.draw(D3js(filename=False), data)
        figure.show()

        # if file name is given
    elif filename is not None:

        # get extension of the file
        extension = filename.split('.')[-1]

        # check if extension is supported
        if extension in figures:
            figure = figures[extension]['fileformat']()

            # check if an other backend is provided
            if backend is not None:
                if backend in backends:
                    _backend = backends[backend]
                else:
                    _backend = figures[extension]['backend']
                    print('The backend "%s" is not available.'
                                'The standard backend was used!', backend)
            else:
                _backend = figures[extension]['backend']

            # draw the figure
            figure.draw(_backend(), data)

            # save the figure
            figure.save(filename)
        else:
            print('Plotting files in the format "%s" is not supported!',
                      extension)
            raise TypeError



class Parser:
    """Object to parse pathpy objects into json like dict."""

    def __init__(self) -> None:
        """Initialize parser object."""
        # initialize variables
        self.figure: defaultdict = defaultdict(dict)
        self.figure['data'] = {}
        self.config: defaultdict = defaultdict(dict)

        self.default_node = {
            'uid': None,
            'node_label': None,
            'node_text': None,
            'node_size': None,
            'node_color': None,
            'node_opacity': None,
            'node_group': None,
            'node_time': None,
            'node_pos': None,
            'node_label_size': None,
            'id_as_label': None,
            'node_style': None,
        }
        self.default_edge = {
            'uid': None,
            'edge_label': None,
            'edge_text': None,
            'edge_size': None,
            'edge_color': None,
            'edge_opacity': None,
            'edge_time': None,
            'source': None,
            'target': None,
            'directed': None,
            'edge_curved': None,
            'edge_style': None,
        }
        self.default_properties = {
            'node': self.default_node,
            'edge': self.default_edge}

        self.default_animation = {
            "enabled": None,
            "begin": None,
            "end": None,
            "steps": None,
            "speed": None,
            "unit": None,
        }

        self.default_config = {
            'animation': self.default_animation
        }

    def __call__(self, obj: Any, plot_config: defaultdict,
                 **kwargs: Any) -> defaultdict:
        """Call the parse function."""
        return self.parse(obj, plot_config, **kwargs)

    @singledispatchmethod
    def parse(self, obj: Any, _config: defaultdict,
              **kwargs: Any) -> defaultdict:
        """Parses the pathpy network into a json like dict."""
        raise NotImplementedError

    @ parse.register(Graph)
    def _parse_static_graph(self, obj: Any, plot_config: defaultdict,
                        temporal=False,
                        **kwargs: Any) -> defaultdict:
        """Parse static network."""

        # update default config
        self.config.update(plot_config)
        # if obj.directed:
        self.config['directed'] = True
        self.config['curved'] = True
        self.config['edge']['directed'] = True

        # convert default units to units
        u2u = UnitConverter(self.config['unit'],
                            kwargs.get('unit', self.config['unit']),
                            dpi=self.config['dpi'])

        self.config['width'] = u2u(self.config['width'])
        self.config['height'] = u2u(self.config['height'])
        self.config['node']['node_size'] = u2u(self.config['node']['node_size'])
        self.config['edge']['edge_size'] = u2u(self.config['edge']['edge_size'])

        # generate default objects
        # iterate over the default config
        for key, values in self.config.items():

            # if objects such as node or edge are in the default config
            if key in self.default_properties:

                # iterate over the attributes
                for attr, value in values.items():

                    # add attributes if the are in the default element
                    if attr in self.default_properties[key]:
                        self.default_properties[key][attr] = value

        # keep default properties
        self.config['default_node'] = self.config['node'].copy()
        self.config['default_edge'] = self.config['edge'].copy()

        # check kwargs and update config
        self.config.update(self.parse_config(self.default_properties, **kwargs))

        # parse layout
        _layout = self.config.get('layout', None)
        if isinstance(_layout, dict):
            self.config['node'].update({'pos': _layout})
            self.config['layout'] = 'euclidean'

        # parse nodes an edges
        nodes = self.parse_nodes(obj, temporal=temporal, **kwargs)
        edges = self.parse_edges(obj, temporal=temporal, **kwargs)

        # convert units to px
        u2px = UnitConverter(self.config['unit'], 'px', dpi=self.config['dpi'])
        for key in ['width', 'height']:
            self.config[key] = u2px(self.config[key])

        nodes = self._convert_size(nodes, u2px, otype='node')
        nodes = self._convert_color(nodes, otype='node')
        edges = self._convert_size(edges, u2px, otype='edge')
        edges = self._convert_color(edges, otype='edge')

        # update layout
        try:
            layout = {n['uid']: n['node_pos'] for n in nodes}
        except KeyError:
            pass
        else:
            nodes = self._update_layout(nodes, layout, u2px)
            self.config['node_pos'] = True

        # add nodes, edges and config to the figure
        self.figure['data']['nodes'] = nodes
        self.figure['data']['edges'] = edges
        self.figure['config'] = self.config

        # return the figure
        return self.figure

    @parse.register(TemporalGraph)
    def _parse_temporal_graph(self, obj: Any, plot_config: defaultdict, **kwargs: Any) -> defaultdict:
        print('Parse a temporal network')

        # get static network to start with
        self._parse_static_graph(obj=obj, plot_config=plot_config,
                           temporal=True, **kwargs)

        # TODO: Fix parse_config for temporal networks
        for key, values in self.parse_config(
                self.default_config, **kwargs).items():
            if isinstance(values, dict):
                for k, v in values.items():
                    print(self.config)
                    print(key)
                    print(k)
                    print(v)
                    self.config[key][k] = v

        # set temporal networkt to true
        self.config['temporal'] = True
        self.figure['data']['changes'] = []

        # get start and end time
        start = obj.start_time
        end = obj.end_time

        # raise error if time frame is not finite
        # if start == float('-inf') or end == float('inf'):
        #     print('The begin/end time is not finite!')
        #     raise ValueError

        # begin and end of the animation
        # TODO: make this more efficient
        animation_start = self.config['animation']['start']
        animation_end = self.config['animation']['end']
        steps = self.config['animation']['steps']
        # if no begin is given take the first observed temporal event
        if animation_start is None:
            animation_start = self._isotime(start)
        elif isinstance(animation_start, (int, float)):
            start = animation_start
            animation_start = self._isotime(animation_start)
        else:
            raise NotImplementedError

        if animation_end is None:
            animation_end = self._isotime(end)
        elif isinstance(animation_end, (int, float)):
            end = animation_end
            animation_end = self._isotime(animation_end)
        else:
            raise NotImplementedError

        # set animation begin and end for d3js
        self.config['animation']['start'] = animation_start
        self.config['animation']['end'] = animation_end

        # get static edges
        # edges = {e['uid']: e for e in self.figure['data']['edges']}

        # _intervals = obj.edges.intervals

        # tree = _intervals.copy()
        # tree.slice(begin)
        # tree.slice(end)
        # tree.remove_envelop(_intervals.begin(), begin)
        # tree.remove_envelop(end, _intervals.end())

        # edge_temp_attr: defaultdict = defaultdict(lambda: defaultdict(dict))
        # for edge in obj.edges.values():
        #     df = edge.attributes.to_frame(history=True)
        #     df = df.where(pd.notnull(df), None)
        #     attr = list(df.to_dict('index').values())
        #     for values in attr:
        #         time = values.pop(TIMESTAMP, None)
        #         edge_temp_attr[edge.uid][time].update(**values)

        def find_nearest(array, value, index=True) -> int:
            value = array[0] if value == float('-inf') else value
            value = array[-1] if value == float('inf') else value
            idx = np.abs(array - value).argmin()
            if index:
                result = int(idx)
            else:
                result = array[idx]
            return result

        # generate temporal edges
        temporal_edges = []
        times = np.linspace(start, end, num=steps)
        i = 0
        for v, w, t in obj.temporal_edges:
            #for event in edge[start:end]:
            _edge: Dict[str, Any] = {}
            _edge['uid'] = str(v)+'-'+str(w)
            _edge['startTime'] = find_nearest(times, t)
            _edge['endTime'] = find_nearest(times, t)
            _edge['active'] = True
            temporal_edges.append(_edge)
            i+=1

        # # get static edges
        # static_edges = {n['uid']: n for n in self.figure['data']['edges']}
        # print(static_edges)
        # _temporal_edges = {}
        # # when edge is active or not
        # for edge in obj.edges[start:end]:
        #     for event in edge[start:end]:
        #         # check if attributes are available
        #         if any(a in event.attributes for
        #                a in self.config['default_edge']):
        #             # get important attributes
        #             _uid = edge.uid
        #             _start = event.attributes.pop('start', start)
        #             _end = event.attributes.pop('end', end)

        #         _edge = {'uid': _uid}
        #         _st = _edge['startTime'] = find_nearest(times, _start)
        #         _et = _edge['endTime'] = find_nearest(times, _end)
        #         _edge['active'] = True
        #         _edge.update(event.attributes)

        #         # add node to the list
        #         key = (_uid, _st, _et)
        #         if key in _temporal_edges:
        #             _temporal_edges[key].update(_edge)
        #         else:
        #             _temporal_edges[key] = _edge

        #         # reset node after attribute change
        #         _edge = {'uid': _uid}
        #         for attr in event.attributes:
        #             if attr in static_edges[_uid]:
        #                 _edge[attr] = static_edges[_uid][attr]

        #         _edge['startTime'] = _et+1
        #         _edge['endTime'] = _et+1
        #         key = (_uid, _et+1, _et+1)
        #         if key in _temporal_edges:
        #             _temporal_edges[key].update(_edge)
        #         else:
        #             _temporal_edges[key] = _edge

        #         temporal_edges.append(_edge)

        # print(_temporal_edges)
        # print(temporal_edges)
        # add temporal edges to the data
        # self.figure['data']['tedges'] = list(_temporal_edges.values())
        self.figure['data']['tedges'] = temporal_edges

        # get static nodes
        static_nodes = {n['uid']: n for n in self.figure['data']['nodes']}

        #temporal_nodes = {}

        # # temporal node attributes
        # for node in obj.nodes[start:end]:
        #     for event in node[start:end]:
        #         # check if attributes are available
        #         if any(a in event.attributes for
        #                a in self.config['default_node']):

        #             # get important attributes
        #             _uid = node.uid
        #             _start = event.attributes.pop('start', start)
        #             _end = event.attributes.pop('end', end)

        #             # generate temporal node
        #             _node = {'uid': _uid}
        #             _st = _node['startTime'] = find_nearest(times, _start)
        #             _et = _node['endTime'] = find_nearest(times, _end)
        #             _node.update(event.attributes)

        #             if _start == float('-inf') and _end == float('inf'):
        #                 continue

        #             # add node to the list
        #             key = (node.uid, _st, _et)
        #             if key in temporal_nodes:
        #                 temporal_nodes[key].update(_node)
        #             else:
        #                 temporal_nodes[key] = _node

        #             # reset node after attribute change
        #             _node = {'uid': _uid}
        #             for attr in event.attributes:
        #                 if attr in static_nodes[_uid]:
        #                     _node[attr] = static_nodes[_uid][attr]

        #             _node['startTime'] = _et
        #             _node['endTime'] = _et
        #             key = (_uid, _et, _et)
        #             if key in temporal_nodes:
        #                 temporal_nodes[key].update(_node)
        #             else:
        #                 temporal_nodes[key] = _node

        self.figure['data']['tnodes'] = list(static_nodes.values())

        # parse layout
        # _layout = self.config.get('layout', None)
        # if isinstance(_layout, dict):
        #     self.config['node'].update({'coordinates': _layout})
        #     self.config['layout'] = 'euclidean'

        # return the figure
        print(self.figure)
        return self.figure

    @ staticmethod
    def _isotime(time: Union[int, float]) -> str:
        """Convert float to ISO 8601 string."""
        return datetime.utcfromtimestamp(time).strftime("%Y-%m-%dT%H:%M:%S")

    def _convert_color(self, objects, otype='node'):
        """Helper function to convert rgb color tuples to JScript color strings"""
        for obj in objects:
            if type(obj['{0}_color'.format(otype)]) == tuple:
                c = 255*np.array(obj['color'])
                obj['{0}_color'.format(otype)] = 'rgb(' + str(int(c[0])) + ', ' + \
                    str(int(c[1])) + ',' + str(int(c[2])) + ')'
            else:
                obj['{0}_color'.format(otype)] = obj['{0}_color'.format(otype)]

        return objects

    def _convert_size(self, objects, converter, otype='node'):
        """Helper function to convert the units of the size of an object."""

        for obj in objects:
            obj['{}_size'.format(otype)] = converter(obj['{0}_size'.format(otype)])

        return objects

    def _update_layout(self, nodes, layout, converter):
        """Helper function to update the layout"""
        # get canvas size and margins
        width = self.config['width']
        height = self.config['height']
        keep_aspect_ratio = self.config['keep_aspect_ratio']

        if self.config['margin'] is None:
            margin = max([float(n['node_size']) for n in nodes])/2+4
        elif isinstance(self.config['margin'], (int, float)):
            margin = converter(self.config['margin'])
        else:
            margin = 0

        margins = {'top': margin, 'left': margin,
                   'bottom': margin, 'right': margin}

        # calculate the scaling ratio
        ratio_x = float('inf')
        ratio_y = float('inf')

        # find min and max values of the points
        min_x = min(layout.items(), key=lambda item: item[1][0])[1][0]
        max_x = max(layout.items(), key=lambda item: item[1][0])[1][0]
        min_y = min(layout.items(), key=lambda item: item[1][1])[1][1]
        max_y = max(layout.items(), key=lambda item: item[1][1])[1][1]

        if max_x-min_x > 0:
            ratio_x = (width-margins['left']-margins['right']) / (max_x-min_x)
        if max_y-min_y > 0:
            ratio_y = (height-margins['top']-margins['bottom']) / (max_y-min_y)

        if keep_aspect_ratio:
            scaling = (min(ratio_x, ratio_y), min(ratio_x, ratio_y))
        else:
            scaling = (ratio_x, ratio_y)

        if scaling[0] == float('inf'):
            scaling = (1, scaling[1])
        if scaling[1] == float('inf'):
            scaling = (scaling[0], 1)

        # apply scaling to the points
        _layout = {}
        for n, (x, y) in layout.items():
            _x = (x)*scaling[0]
            _y = (y)*scaling[1]
            _layout[n] = (_x, _y)

        # find min and max values of new the points
        min_x = min(_layout.items(), key=lambda item: item[1][0])[1][0]
        max_x = max(_layout.items(), key=lambda item: item[1][0])[1][0]
        min_y = min(_layout.items(), key=lambda item: item[1][1])[1][1]
        max_y = max(_layout.items(), key=lambda item: item[1][1])[1][1]

        # calculate the translation
        translation = (((width-margins['left']-margins['right'])/2
                        + margins['left']) - ((max_x-min_x)/2 + min_x),
                       ((height-margins['top']-margins['bottom'])/2
                       + margins['bottom']) - ((max_y-min_y)/2 + min_y))

        # apply translation to the points
        for n, (x, y) in _layout.items():
            _x = (x)+translation[0]
            _y = (y)+translation[1]
            _layout[n] = (_x, _y)

        for node in nodes:
            node['node_pos'] = _layout[node['uid']]

        return nodes

    def parse_config(self, properties: dict, **kwargs: Any) -> defaultdict:
        """Parse the config file."""

        # initialize temporal dict
        _config: defaultdict = defaultdict(dict)

        # extend default dict
        for key in properties:
            _config[key] = defaultdict(dict)

        # iterate over kwargs
        for key, value in kwargs.items():

            # split key from kwargs
            _key = key.split("_", 1)

            # check if key is valid
            if _key[0] in properties:
                if _key[1] in properties[_key[0]]:

                    # add value to dictionary
                    _config[_key[0]][_key[1]] = value

            # check if key is in the default config
            elif key in self.config:
                _config[key] = value

        return _config

    def parse_nodes(self, graph, temporal=False, **kwargs) -> List:
        """Parse nodes"""

        # initialize temporal dict
        node_dict: defaultdict = defaultdict(dict)

        # get mapping if defined
        mapping = kwargs.get('mapping', None)

        # iterate over nodes
        for node in graph.nodes:

            # add default properties to nodes
            node_dict[node] = self.default_properties['node'].copy()

            # add node ids
            node_dict[node]['uid'] = node

            # add node attributes
            for attr in graph.data.node_attrs():

                # if mapping is given map the attribute
                if mapping is not None and attr in mapping:
                    attr = mapping[attr]

                # check if attribute is in the default object
                if attr in self.default_properties['node']:
                    # update attribute
                    node_dict[node][attr] = graph[attr, node]

        # update objects based on the kwargs
        # iterate over the kwargs config
        for key, values in self.config['node'].items():

            # check if new attribute is a single object
            if isinstance(values, (str, int, float, bool)):
                for obj in node_dict.values():
                    obj.update({key: values})

            # check if new attribute is a list
            elif isinstance(values, list):
                for i, obj in enumerate(node_dict.values()):
                    try:
                        obj[key] = values[i]
                    except KeyError:
                        pass

            # check if new attribute is a dict
            elif isinstance(values, dict):
                for k in node_dict:
                    if k in values:
                        node_dict[k][key] = values[k]
            # otherwise raise error
            else:
                print('Something went wrong, by formatting the values!')
                raise ValueError

        # remove None values from the objects
        for key, values in node_dict.items():
            for attr, value in list(values.items()):
                if value is None:
                    node_dict[key].pop(attr)

        return list(node_dict.values())

    def parse_edges(self, graph, temporal=False, **kwargs) -> List:
        """Parse edges."""

        # initialize temporal dict
        edge_dict: defaultdict = defaultdict(dict)

        # get mapping if defined
        mapping = kwargs.get('mapping', None)

        # iterate over edges
        for u, v in graph.edges:

            # add default properties to the obj
            uid = str(u)+'-'+str(v)
            edge_dict[uid] = self.default_properties['edge'].copy()

            # add obj uid
            edge_dict[uid]['uid'] = uid

            # if obj is an edge add source and target nodes
            edge_dict[uid]['source'] = u
            edge_dict[uid]['target'] = v

            # add obj attributes
            for attr in graph.data.edge_attrs():
                # if mapping is given map the attribute
                if mapping is not None and attr in mapping:
                    attr = mapping[attr]

                # check if attribute is in the default object
                if attr in self.default_properties['edge']:

                    # update attribute if given
                    edge_dict[uid][attr] = graph[attr, u, v]

        # update objects based on the kwargs
        # iterate over the kwargs config
        for key, values in self.config['edge'].items():

            # check if new attribute is a single object
            if isinstance(values, (str, int, float, bool)):
                for obj in edge_dict.values():
                    obj.update({key: values})

            # check if new attribute is a list
            elif isinstance(values, list):
                for i, obj in enumerate(edge_dict.values()):
                    try:
                        obj[key] = values[i]
                    except KeyError:
                        pass

            # check if new attribute is a dict
            elif isinstance(values, dict):
                for k in edge_dict:
                    if k in values:
                        edge_dict[k][key] = values[k]
            # otherwise raise error
            else:
                print('Something went wrong, by formatting the values!')
                raise ValueError

        # remove None values from objects
        for key, values in edge_dict.items():
            for attr, value in list(values.items()):
                if value is None:
                    edge_dict[key].pop(attr)

        return list(edge_dict.values())

# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
