from __future__ import annotations
from typing import Any, List, Optional, Union, Dict, Callable
from collections import defaultdict
from singledispatchmethod import singledispatchmethod  # remove for python 3.8

import os
import uuid
import json

from string import Template

from pathpyG import config
from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph


def html_static_plot(graph: Graph):
    data = parse_static_graph(graph)

   # generate unique dom uids
    widgets_id = 'x'+uuid.uuid4().hex
    network_id = 'x'+uuid.uuid4().hex

    # template directory
    template_dir = str(os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        os.path.normpath('visualisations/templates/network')))

    template_file = os.path.join(template_dir, 'template_vscode.html')

    with open(template_file) as f:
        template = f.read()

        # load css template
        css_file = os.path.join(template_dir, 'css/style.css')

        with open(css_file) as f2:
            css_template = f2.read()

        # Substitute parameters in template
        js = Template(template).substitute(divId=widgets_id,
                                              svgId=network_id,
                                              config=json.dumps({}),
                                              data=json.dumps(data))

        # generate html file with css styles
        html = '<style>\n' + css_template + '\n</style>\n'

        # if config['environment']['IDE'] != 'vs code':
        #     html = html + '<script charset="utf-8" src="https://d3js.org/d3.v5.min.js"></script>\n <script charset="utf-8" src="https://requirejs.org/docs/release/2.3.6/minified/require.js"></script>'
        # div environment for the widgets and network
        html = html + \
            '<div id="{}"></div>\n<div id="{}"></div>\n'.format(
                widgets_id, network_id)

        # add js code
        html = html+js

    from IPython.display import display, HTML
    display(HTML(html))

    return html


def html_temporal_plot(graph: TemporalGraph):
    data = parse_temporal_graph(graph)

def parse_static_graph(graph):

    data = {}
    data['nodes'] = {}
    data['edges'] = {}

    # iterate over nodes
    for node in graph.nodes:

        id = str(node)
        # add node id
        data['nodes'][id] = {}
        data['nodes'][id]['id'] = id

        # add default properties to nodes
        for prop in config['node']:
            data['nodes'][id][prop] = config['node'][prop]

        # add node attributes
        for attr in graph.node_attrs():
            data['nodes'][id][attr.replace('node_', '')] = graph[attr, node]

     # iterate over edges
    for u, v in graph.edges:

        id = str(u)+'-'+str(v)

        data['edges'][id] = {}

        # add default properties to edge
        for prop in config['edge']:
            data['edges'][id][prop] = config['edge'][prop]

        # add edge uid
        data['edges'][id]['id'] = id

        # if obj is an edge add source and target nodes
        data['edges'][id]['source'] = str(u)
        data['edges'][id]['target'] = str(v)

        # add obj attributes
        for attr in graph.edge_attrs():
            data['edges'][id][attr.replace('edges_', '')] = graph[attr, u, v]
    return data