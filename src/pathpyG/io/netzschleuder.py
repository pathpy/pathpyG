from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import json
import pickle
import struct
from collections import defaultdict
from urllib import request
from urllib.error import HTTPError
import tempfile
import zipfile
from io import BytesIO

from numpy import array
import torch
import pandas as pd

from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph
from pathpyG.utils.config import config
from pathpyG.io.pandas import df_to_graph, df_to_temporal_graph
from pathpyG.io.graphtool import parse_graphtool_format
from pathpyG.io.pandas import read_csv_graph, read_csv_temporal_graph, add_node_attributes, add_edge_attributes


def list_netzschleuder_records(base_url: str = 'https://networks.skewed.de', **kwargs: Any) -> Union[list, dict]:
    """
    Read a list of data sets available at the netzschleuder repository.

    Args:
        base_url: Base URL of netzschleuder repository
        **kwargs: Keyword arguments that will be passed to the netzschleuder repository as HTTP GET parameters.
            For supported parameters see https://networks.skewed.de/api


    Examples:
        Return a list of all data sets

        >>> import pathpyG as pp
        >>> pp.io.list_netzschleuder_records()
        ['karate', 'reality_mining', 'sp_hypertext', ...]

        Return a list of all data sets with a given tag

        >>> pp.io.list_netzschleuder_records(tags='temporal')
        ['reality_mining', 'sp_hypertext', ...]

        Return a dictionary containing all data set names (keys) as well as all network attributes

        >>> pp.io.list_netzschleuder_records(full=True)
        { 'reality_mining': [...], 'karate': [...] }


    Returns:
        Either a list of data set names or a dictionary containing all data set names and network attributes.

    """
    url = '/api/nets'
    for k, v in kwargs.items():
        url += '?{0}={1}'.format(k, v)
    try:
        f = request.urlopen(base_url + url).read()
        return json.loads(f)
    except HTTPError:
        msg = 'Could not connect to netzschleuder repository at {0}'.format(base_url)
        # LOG.error(msg)
        raise Exception(msg)



def read_netzschleuder_record(name: str, base_url: str = 'https://networks.skewed.de') -> dict:
    """
    Read metadata of a single data record with given name from the netzschleuder repository

    Args:
        name: Name of the data set for which to retrieve the metadata
        base_url: Base URL of netzschleuder repository

    Examples:
        Retrieve metadata of karate club network
        
        >>> import pathpyG as pp
        >>> metdata = pp.io.read_netzschleuder_record('karate')
        >>> print(metadata)
        {
            'analyses': {'77': {'average_degree': 4.52... } }
        }

    Returns:
        Dictionary containing key-value pairs of metadata
    """
    url = '/api/net/{0}'.format(name)
    try:
        return json.loads(request.urlopen(base_url + url).read())
    except HTTPError:
        msg = 'Could not connect to netzschleuder repository at {0}'.format(base_url)
        #LOG.error(msg)
        raise Exception(msg)


def read_netzschleuder_graph(name: str, net: Optional[str] = None, multiedges: bool = False,
        base_url: str='https://networks.skewed.de', format='csv') -> Graph:
    """Read a pathpyG graph or temporal graph from the netzschleuder repository.

    Args:
        name: Name of the network data set to read from
        net: Identifier of the network within the data set to read. For data sets
            containing a single network only, this can be set to None.
        ignore_temporal: If False, this function will return a static or temporal network depending
            on whether edges contain a time attribute. If True, pathpy will not interpret
            time attributes and thus always return a static network.
        base_url: Base URL of netzschleuder repository
        format: for 'csv' a zipped csv file will be downloaded, for 'gt' the binary graphtool format will be retrieved via the API

    Examples:
        Read network '77' from karate club data set

        >>> import pathpyG as pp
        >>> n = pp.io.read_netzschleuder_network('karate', '77')
        >>> print(type(n))
        >>> pp.plot(n)
        pp.Graph


    Returns:
        an instance of Graph

    """
 # build URL

    
    # retrieve properties of data record via API
    properties = json.loads(request.urlopen(f'{base_url}/api/net/{name}').read())
    # print(properties)

    timestamps = 'Timestamps' in properties['tags']

    if not net:
        analyses = properties['analyses']
        net = name
    else:
        analyses = properties['analyses'][net]         
    
    is_directed = analyses['is_directed']
    num_nodes = analyses['num_vertices']
    
    if format == 'csv': 
        url = f'{base_url}/net/{name}/files/{net}.csv.zip'
        # print(url)
        try:
            response = request.urlopen(url)
            # decompress zip into temporary folder
            data = BytesIO(response.read())

            with zipfile.ZipFile(data, 'r') as zip_ref:
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_ref.extractall(path=temp_dir)
                    print(temp_dir)

                    # the gprop file contains lines with property name/value pairs 
                    # gprops = pd.read_csv(f'{temp_dir}/gprops.csv', header=0, sep=',', skip_blank_lines=True, skipinitialspace=True)
                    # nodes.csv contains node indices with node properties (like name)                    
                    edges = pd.read_csv(f'{temp_dir}/edges.csv', header=0, sep=',', skip_blank_lines=True, skipinitialspace=True)

                    # rename columns
                    edges.rename(columns={'# source': 'v', 'target': 'w'}, inplace=True)                    
                    if timestamps:
                        edges.rename(columns={'time': 't'}, inplace=True)                    
                    # print(edges)

                    if timestamps:
                        g = df_to_temporal_graph(df=edges, multiedges=True, is_undirected=not is_directed, num_nodes=num_nodes)
                    else:
                        g = df_to_graph(df=edges, multiedges=True, num_nodes=num_nodes)
                        if not is_directed:
                            g = g.to_undirected()


                    node_attrs = pd.read_csv(f'{temp_dir}/nodes.csv', header=0, sep=',', skip_blank_lines=True, skipinitialspace=True)
                    node_attrs.rename(columns={'# index': 'index'}, inplace=True)
                    # print(node_attrs)
                    #print(set(list(node_attrs['v'].astype(str))))
                    #print(set([v for v in g.nodes]))
                    add_node_attributes(node_attrs, g)
                    add_edge_attributes(edges, g)

                    for x in analyses:
                        g.data[x] = analyses[x]

                    return g
            # g = read_csv_graph(edges_file, sep=',', header=True, is_undirected = undirected, multiedges=True)

            # add_node_attributes(node_attr, g)
        except HTTPError:
            msg = 'Could not connect to netzschleuder repository at {0}'.format(base_url)
            raise Exception(msg)

    elif format == 'gt':
        try:
            import zstandard as zstd

            url = '/net/{0}/files/{1}.gt.zst'.format(name, net)
            try:
                f = request.urlopen(base_url + url)
                # decompress data
                dctx = zstd.ZstdDecompressor()
                reader = dctx.stream_reader(f)
                decompressed = reader.readall()

                # parse graphtool binary format
                return parse_graphtool_format(bytes(decompressed))
            except HTTPError:
                msg = 'Could not connect to netzschleuder repository at {0}'.format(base_url)
                raise Exception(msg)
        except ModuleNotFoundError:
            msg = 'Package zstandard is required to decompress graphtool files. Please install module, e.g., using "pip install zstandard.'
            # LOG.error(msg)
            raise Exception(msg)
