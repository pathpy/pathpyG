from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import json
import pickle
import struct
from collections import defaultdict
from urllib import request
from urllib.error import HTTPError

from numpy import array
import torch

from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph
from pathpyG.utils.config import config


# TODO: Update documentation! Some of the classes mentioned below do not exist anymore.
def _parse_property_value(data: bytes, ptr: int, type_index: int, endianness: str) -> Tuple[Optional[Any], int]:
    """
    Parse a property value as well as the number of processed bytes.

    Args:
        data: byte array containing the data to be decoded
        ptr: index of the first byte to be parsed
        type_index: integer representing the type of the property value to be parsed
        endianness: string representation of endianness, where `>` represents Big Endian
        and `<` represents Little Endian

    Returns:
        Tuple $(v, n)$ consisting of the property value $v$ and the number of bytes $n$ processed
    """
    if type_index == 0:
        return (bool(data[ptr]), 1)
    elif type_index == 1:
        return (struct.unpack(endianness + 'h', data[ptr:ptr+2])[0], 2)
    elif type_index == 2:
        return (struct.unpack(endianness + 'i', data[ptr:ptr+4])[0], 4)
    elif type_index == 3:
        return (struct.unpack(endianness + 'q', data[ptr:ptr+8])[0], 8)
    elif type_index == 4:
        return (struct.unpack(endianness + 'd', data[ptr:ptr+8])[0], 8)
    elif type_index == 5:
        print('pathpy does not support properties with type long double. Properties have been dropped.')
        return (None, 16)
    elif type_index == 6:
        str_len = struct.unpack(endianness + 'Q', data[ptr:ptr+8])[0]
        str = data[ptr+8:ptr+8+str_len].decode('utf-8')
        return (str, 8 + str_len)
    elif type_index == 7:
        num_values = struct.unpack(endianness + 'Q', data[ptr:ptr+8])[0]
        offset = 8
        vals = []
        for i in range(num_values):
            vals.append(bool(data[ptr+offset:ptr+offset+1]))
            offset += 1
        return (array(vals), 8 + num_values)
    elif type_index == 8:
        num_values = struct.unpack(endianness + 'Q', data[ptr:ptr+8])[0]
        offset = 8
        vals = []
        for i in range(num_values):
            vals.append(struct.unpack(endianness + 'h', data[ptr+offset:ptr+offset+2])[0])
            offset += 4
        return (array(vals), 8 + 2*num_values)
    elif type_index == 9:
        num_values = struct.unpack(endianness + 'Q', data[ptr:ptr+8])[0]
        offset = 8
        vals = []
        for i in range(num_values):
            vals.append(struct.unpack(endianness + 'i', data[ptr+offset:ptr+offset+4])[0])
            offset += 4
        return (array(vals), 8 + 4*num_values)
    elif type_index == 10:
        num_values = struct.unpack(endianness + 'Q', data[ptr:ptr+8])[0]
        offset = 8
        vals = []
        for i in range(num_values):
            vals.append(struct.unpack(endianness + 'Q', data[ptr+offset:ptr+offset+8])[0])
            offset += 8
        return (None, 8 + 8*num_values)
    elif type_index == 11:
        num_values = struct.unpack(endianness + 'Q', data[ptr:ptr+8])[0]
        offset = 8
        vals = []
        for i in range(num_values):
            vals.append(struct.unpack(endianness + 'd', data[ptr+offset:ptr+offset+8])[0])
            offset += 8
        return (array(vals), 8 + 8*num_values)
    elif type_index == 12:
        val_len = struct.unpack(endianness + 'Q', data[ptr:ptr+8])[0]
        print('pathpyG does not support properties with type vector<long double>. Properties have been dropped.')
        return (None, 8 + 16*val_len)
    elif type_index == 13:
        num_strings = struct.unpack(endianness + 'Q', data[ptr:ptr+8])[0]
        offset = 8
        strs = []
        for i in range(num_strings):
            str_len = struct.unpack(endianness + 'Q', data[ptr+offset:ptr+offset+8])[0]
            offset += 8
            strs.append(data[ptr+offset:ptr+offset+str_len].decode('utf-8'))
            offset += str_len

        return (strs, offset)
    elif type_index == 14:
        val_len = struct.unpack(endianness + 'Q', data[ptr:ptr+8])[0]
        return (pickle.loads(data[ptr+8:ptr+8+val_len]), 8 + val_len)
    else:
        msg = 'Unknown type index {0} while parsing graphtool file'.format(type_index)
        print(msg)
        raise Exception(msg)


def parse_graphtool_format(data: bytes, id_node_attr=None) -> Graph:
    """
    Decodes data in graphtool binary format and returns a [`Graph`][pathpyG.Graph]. For a documentation of
    hte graphtool binary format, see see doc at https://graph-tool.skewed.de/static/doc/gt_format.html

    Args:
        data: Array of bys to be decoded
        ignore_temporal: If False, this function will return a static or temporal network depending
            on whether edges contain a time attribute. If True, pathpy will not interpret
            time attributes and thus always return a static network.

    Returns:
        Network or TemporalNetwork: a static or temporal network object
    """

    # check magic bytes
    if data[0:6] != b'\xe2\x9b\xbe\x20\x67\x74':
        print('Invalid graphtool file. Wrong magic bytes.')
        raise Exception('Invalid graphtool file. Wrong magic bytes.')
    ptr = 6

    # read graphtool version byte
    graphtool_version = int(data[ptr])
    ptr += 1

    # read endianness
    if bool(data[ptr]):
        graphtool_endianness = '>'
    else:
        graphtool_endianness = '<'
    ptr += 1

    # read length of comment
    str_len = struct.unpack(graphtool_endianness + 'Q', data[ptr:ptr+8])[0]
    ptr += 8

    # read string comment
    comment = data[ptr:ptr+str_len].decode('ascii')
    ptr += str_len

    # read network directedness
    directed = bool(data[ptr])
    ptr += 1

    # read number of nodes
    n_nodes = struct.unpack(graphtool_endianness + 'Q', data[ptr:ptr+8])[0]
    ptr += 8

    # create pandas dataframe
    network_dict = {}
    # n = Network(directed = directed, multiedges=True)

    # determine binary representation of neighbour lists
    if n_nodes<2**8:
        fmt = 'B'
        d = 1
    elif n_nodes<2**16:
        fmt = 'H'
        d = 2
    elif n_nodes<2**32:
        fmt = 'I'
        d = 4
    else:
        fmt = 'Q'
        d = 8

    sources = []
    targets = []
    # parse lists of out-neighbors for all n nodes
    n_edges = 0
    for v in range(n_nodes):
        # read number of neighbors
        num_neighbors = struct.unpack(graphtool_endianness + 'Q', data[ptr:ptr+8])[0]
        ptr += 8

        # add edges to record
        for j in range(num_neighbors):
            w = struct.unpack(graphtool_endianness + fmt, data[ptr:ptr+d])[0]
            ptr += d
            sources.append(v)
            targets.append(w)
            n_edges += 1

    #network_data = pd.DataFrame.from_dict(network_dict, orient='index', columns=['v', 'w'])

    # collect attributes from property maps
    graph_attr = dict()
    node_attr = dict()
    edge_attr = dict()

    # parse property maps
    property_maps = struct.unpack(graphtool_endianness + 'Q', data[ptr:ptr+8])[0]
    ptr += 8

    for i in range(property_maps):
        key_type = struct.unpack(graphtool_endianness + 'B', data[ptr:ptr+1])[0]
        ptr += 1

        property_len  = struct.unpack(graphtool_endianness + 'Q', data[ptr:ptr+8])[0]
        ptr += 8

        property_name = data[ptr:ptr+property_len].decode('ascii')
        ptr += property_len

        property_type = struct.unpack(graphtool_endianness + 'B', data[ptr:ptr+1])[0]
        ptr += 1

        if key_type == 0: # graph-level property
            res = _parse_property_value(data, ptr, property_type, graphtool_endianness)
            graph_attr[property_name] = res[0]
            ptr += res[1]
        elif key_type == 1: # node-level property
            if property_name not in node_attr:
                node_attr[property_name] = []
            for v in range(n_nodes):
                res = _parse_property_value(data, ptr, property_type, graphtool_endianness)
                node_attr[property_name].append([res[0]])
                ptr += res[1]
        elif key_type == 2: # edge-level property
            if property_name not in edge_attr:
                edge_attr[property_name] = []
            for e in range(n_edges):
                res = _parse_property_value(data, ptr, property_type, graphtool_endianness)
                edge_attr[property_name].append(res[0])
                ptr += res[1]
        else:
            print('Unknown key type {0}'.format(key_type))

    # LOG.info('Version \t= {0}'.format(graphtool_version))
    # LOG.info('Endianness \t= {0}'.format(graphtool_endianness))
    # LOG.info('comment size \t= {0}'.format(str_len))
    # LOG.info('comment \t= {0}'.format(comment))
    # LOG.info('directed \t= {0}'.format(directed))
    # LOG.info('nodes \t\t= {0}'.format(n_nodes))

    # add edge properties to data frame
    # for p in edge_attribute_names:
    #     # due to use of default_dict, this will add NA values to edges which have missing properties
    #     network_data[p] = [ edge_attributes[e][p] for e in range(n_edges) ]

    # create graph from pandas dataframe


    # if 'time' in edge_attribute_names and not ignore_temporal:
    #     raise Exception('')
    #     n = to_temporal_network(network_data, directed=directed, **network_attributes)
    # else:


    if id_node_attr:
        node_id = node_attr[id_node_attr]
    else:
        node_id = []

    g = Graph(edge_index=torch.tensor([sources, targets], dtype=torch.long).to(config['torch']['device']), node_id=node_id)
    for a in node_attr:
        if not a.startswith('node_'):
            # print(node_attr[a])
            # g.data['node_{0}'.format(a)] = torch.tensor(node_attr[a], dtype=torch.float).to(config['torch']['device'])
            g.data['node_{0}'.format(a)] = node_attr[a]
    for a in edge_attr:
        if not a.startswith('edge_'):
            g.data['edge_{0}'.format(a)] = torch.tensor(edge_attr[a], dtype=torch.float).to(config['torch']['device'])
    for a in graph_attr:
        g.data[a] = graph_attr[a]
    return g


    # for v in node_attributes:
    #     for p in node_attributes[v]:
    #         # for now we remove _pos for temporal networks due to type being incompatible with plotting
    #         if p != '_pos' or ('time' not in edge_attribute_names or ignore_temporal):
    #             n.nodes[v][p] = node_attributes[v][p]


def read_graphtool(file: str, ignore_temporal: bool=False, multiedges: bool=False) -> Optional[Union[Graph, TemporalGraph]]:
    """
    Read a file in graphtool binary format.

    Args:
        file: Path to graphtool file to be read
    """
    with open(file, 'rb') as f:
        if '.zst' in file:
            try:
                import zstandard as zstd
                dctx = zstd.ZstdDecompressor()
                data = f.read()
                return parse_graphtool_format(dctx.decompress(data, max_output_size=len(data)))
            except ModuleNotFoundError:
                msg = 'Package zstandard is required to decompress graphtool files. Please install module, e.g., using "pip install zstandard".'
                # LOG.error(msg)
                raise Exception(msg)
        else:
            return parse_graphtool_format(f.read(), multiedges)



def list_netzschleuder_records(base_url: str='https://networks.skewed.de', **kwargs: Any) -> Union[list, dict]:
    """
    Read a list of data sets available at the netzschleuder repository.

    Args:
        base_url: Base URL of netzschleuder repository
        **kwargs: Keyword arguments that will be passed to the netzschleuder repository as HTTP GET parameters.
            For supported parameters see https://networks.skewed.de/api


    Examples:
        Return a list of all data sets

        >>> import pathpy as pp
        >>> pp.io.graphtool.list_netzschleuder_records()
        ['karate', 'reality_mining', 'sp_hypertext', ...]

        Return a list of all data sets with a given tag

        >>> pp.io.graphtool.list_netzschleuder_records(tags='temporal')
        ['reality_mining', 'sp_hypertext', ...]

        Return a dictionary containing all data set names (keys) as well as all network attributes

        >>> pp.io.graphtool.list_netzschleuder_records(full=True)
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



def read_netzschleuder_record(name: str, base_url: str='https://networks.skewed.de') -> dict:
    """
    Read metadata of a single data record with given name from the netzschleuder repository

    Args:
        name: Name of the data set for which to retrieve the metadata
        base_url: Base URL of netzschleuder repository

    Examples:
        Retrieve metadata of karate club network
        
        >>> import pathpy as pp
        >>> metdata = pp.io.graphtool.read_netzschleuder_record('karate')
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


def read_netzschleuder_network(name: str, net: Optional[str]=None,
        ignore_temporal: bool=False, multiedges: bool=False,
        base_url: str='https://networks.skewed.de') -> Union[Graph, TemporalGraph]:
    """Read a pathpy network record from the netzschleuder repository.

    Args:
        name: Name of the network data sets to read from
        net: Identifier of the network within the data set to read. For data sets
            containing a single network only, this can be set to None.
        ignore_temporal: If False, this function will return a static or temporal network depending
            on whether edges contain a time attribute. If True, pathpy will not interpret
            time attributes and thus always return a static network.
        base_url: Base URL of netzschleuder repository

    Examples:
        Read network '77' from karate club data set

        >>> import pathpy as pp
        >>> n = pp.io.graphtool.read_netzschleuder_network('karate', '77')
        >>> print(type(n))
        >>> pp.plot(n)
        pp.Network

        Read a temporal network from a data set containing a single network only
        (i.e. net can be omitted):

        >>> n = pp.io.graphtool.read_netzschleuder_network('reality_mining')
        >>> print(type(n))
        >>> pp.plot(n)
        pp.TemporalNetwork

        Read temporal network but ignore time attribute of edges:

        >>> n = pp.io.graphtool.read_netzschleuder_network('reality_mining', ignore_temporal=True)
        >>> print(type(n))
        >>> pp.plot(n)
        pp.Network


    Returns:
        Depending on whether the network data set contains an edge attribute
        `time` (and whether ignore_temporal is set to True), this function
        returns an instance of Network or TemporalNetwork

    """
    try:
        import zstandard as zstd

        # retrieve network properties
        url = '/api/net/{0}'.format(name)
        properties = json.loads(request.urlopen(base_url + url).read())

        # retrieve data
        if not net:
            net = name
        url = '/net/{0}/files/{1}.gt.zst'.format(name, net)
        try:
            f = request.urlopen(base_url + url)
        except HTTPError:
            msg = 'Could not connect to netzschleuder repository at {0}'.format(base_url)
            #LOG.error(msg)
            raise Exception(msg)

        # decompress data
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        decompressed = reader.readall()

        # parse graphtool binary format
        return parse_graphtool_format(bytes(decompressed))

    except ModuleNotFoundError:
        msg = 'Package zstandard is required to decompress graphtool files. Please install module, e.g., using "pip install zstandard.'
        # LOG.error(msg)
        raise Exception(msg)
