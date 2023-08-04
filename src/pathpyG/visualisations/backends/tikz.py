"""Backend for tikz."""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : tikz.py -- Module to draw a tikz-network
# Author    : Jürgen Hackl <hackl@ifi.uzh.ch>
# Time-stamp: <Tue 2020-05-05 14:26 juergen>
#
# Copyright (c) 2016-2019 Pathpy Developers
# =============================================================================
from __future__ import annotations  # remove for python 3.8
from random import uniform

from pathpyG.visualisations.utils import UnitConverter, bend_factor



class Tikz:
    """Class to draw tikz objects."""

    def __init__(self) -> None:
        """Initialize tikz drawer"""
        self.default_node_kwargs = {
            'node_size': 'size',
            'node_color': 'color',
            'node_opacity': 'opacity',
            'node_label': 'label',
            'node_label_position': 'position',
            'node_label_distance': 'distance',
            'node_label_color': 'fontcolor',
            'node_label_size': 'fontscale',
            'node_shape': 'shape',
            'node_style': 'style',
            'node_layer': 'layer',
        }
        self.default_node_args = {
            'label_off': 'NoLabel',
            'id_as_label': 'IdAsLabel',
            'math_mode': 'Math',
            'rgb': 'RGB',
            'pseudo': 'Pseudo',
        }
        self.default_edge_kwargs = {
            'edge_size': 'lw',
            'edge_color': 'color',
            'edge_opacity': 'opacity',
            'edge_curved': 'bend',
            'edge_label': 'label',
            'edge_label_position': 'position',
            'edge_label_distance': 'distance',
            'edge_label_color': 'fontcolor',
            'edge_label_size': 'fontscale',
            'edge_style': 'style',
            'edge_loopsize': 'loopsize',
            'edge_loopposition': 'loopposition',
            'edge_loopshape': 'loopshape',
        }
        self.default_edge_args = {
            'directed': 'Direct',
            'math_mode': 'Math',
            'rgb': 'RGB',
            'not_in_bg': 'NotInBG',
        }
        self.default_kwargs = {
            'nodes': self.default_node_kwargs,
            'edges': self.default_edge_kwargs,
        }

        self.default_args = {
            'nodes': self.default_node_args,
            'edges': self.default_edge_args,
        }

    def to_tex(self, figure) -> str:
        """Convert figure to a single html document."""
        print('Generate single tex document.')

        # clean config
        config = figure['config']
        config.pop('node')
        config.pop('edge')

        # initialize unit converters
        px2cm = UnitConverter('px', 'cm')
        px2pt = UnitConverter('px', 'pt')

        for key in ['width', 'height']:
            config[key] = px2cm(config[key])

        # clean data
        data = figure['data']

        for node in data['nodes']:
            node['node_size'] = px2cm(node['node_size'])
        for edge in data['edges']:
            edge['edge_size'] = px2pt(edge['edge_size'])
            if edge.get('edge_curved', None) is not None:
                if not config.get('edge_curved', False):
                    edge.pop('edge_curved')
                else:
                    edge['edge_curved'] = bend_factor(edge['edge_curved'])
            if not config.get('directed', False):
                edge.pop('directed', None)

        tex = ''
        for element in ['nodes', 'edges']:
            for obj in data[element]:
                if element == 'nodes':
                    _xy = obj.get('edge_pos', None)
                    if _xy is None:
                        _xy = (uniform(0, config['width']),
                               uniform(0, config['height']))
                    else:
                        _xy = (px2cm(_xy[0]), px2cm(_xy[1]))
                    string = '\\Vertex[x={x:.{n}f},y={y:.{n}f}' \
                        ''.format(x=_xy[0], y=_xy[1], n=3)
                else:
                    string = '\\Edge['

                for key, value in self.default_kwargs[element].items():
                    if key in obj:
                        string += ',{}={}'.format(value, obj[key])

                for key, value in self.default_args[element].items():
                    if key in obj and obj[key] is True:
                        string += ',{}'.format(value)

                if element == 'nodes':
                    string += ']{{{}}}\n'.format(obj['uid'])
                else:
                    string += ']({})({})\n'.format(obj['source'],
                                                   obj['target'])
                tex += string
        return tex

    def to_csv(self, figure) -> str:
        """Convert figure to a single html document."""
        print('Generate csv documents.')
        raise NotImplementedError

# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
