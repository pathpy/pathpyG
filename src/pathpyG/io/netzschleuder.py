"""PathpyG input/output module for the netzschleuder repository."""

import json
import tempfile
import zipfile
from io import BytesIO
from typing import Any, Optional, Union
from urllib import request
from urllib.error import HTTPError

import pandas as pd

from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.io.pandas import add_node_attributes, df_to_graph, df_to_temporal_graph


def list_netzschleuder_records(base_url: str = "https://networks.skewed.de", **kwargs: Any) -> Union[list, dict]:
    """Read a list of data sets available at the netzschleuder repository.

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

        >>> pp.io.list_netzschleuder_records(tags="temporal")
        ['reality_mining', 'sp_hypertext', ...]

        Return a dictionary containing all data set names (keys) as well as all network attributes

        >>> pp.io.list_netzschleuder_records(full=True)
        { 'reality_mining': [...], 'karate': [...] }

    Returns:
        Either a list of data set names or a dictionary containing all data set names and network attributes.
    """
    url = "/api/nets"
    for k, v in kwargs.items():
        url += "?{0}={1}".format(k, v)
    try:
        f = request.urlopen(base_url + url).read()
        return json.loads(f)
    except HTTPError:
        msg = "Could not connect to netzschleuder repository at {0}".format(base_url)
        # LOG.error(msg)
        raise Exception(msg)


def read_netzschleuder_record(name: str, base_url: str = "https://networks.skewed.de") -> dict:
    """Read metadata of a single data record with given name from the netzschleuder repository.

    Args:
        name: Name of the data set for which to retrieve the metadata
        base_url: Base URL of netzschleuder repository

    Examples:
        Retrieve metadata of karate club network

        >>> import pathpyG as pp
        >>> metdata = pp.io.read_netzschleuder_record("karate")
        >>> print(metadata)
        {
            'analyses': {'77': {'average_degree': 4.52... } }
        }

    Returns:
        Dictionary containing key-value pairs of metadata
    """
    url = f"/api/net/{name}"
    try:
        return json.loads(request.urlopen(base_url + url).read())
    except HTTPError as exc:
        msg = f"Could not connect to netzschleuder repository at {base_url}"
        # LOG.error(msg)
        raise Exception(msg) from exc


def read_netzschleuder_graph(
    name: str,
    network: Optional[str] = None,
    multiedges: bool = False,
    time_attr: Optional[str] = None,
    base_url: str = "https://networks.skewed.de",
) -> Union[Graph, TemporalGraph]:
    """Read a graph or temporal graph from the netzschleuder repository.

    Args:
        name: Name of the network data set to read from
        network: Identifier of the network within the data set to read. For data sets
            containing a single network only, this can be set to None.
        multiedges: Whether to allow multiedges in the constructed graph
        time_attr: Name of the edge attribute containing time stamps. If None,
            the function will read the graph as static network.
        base_url: Base URL of netzschleuder repository.

    Examples:
        Read network '77' from karate club data set

        >>> import pathpyG as pp
        >>> n = pp.io.read_netzschleuder_network(name="karate", network="77")
        >>> print(type(n))
        >>> pp.plot(n)
        pp.Graph

    Returns:
        Graph or TemporalGraph object
    """
    # build URL
    try:
        # retrieve properties of data record via API
        properties = json.loads(request.urlopen(f"{base_url}/api/net/{name}").read())

        timestamps = time_attr is not None

        if not network:
            analyses = properties["analyses"]
            network = name
        else:
            analyses = properties["analyses"][network]

        try:
            is_directed = analyses["is_directed"]
            num_nodes = analyses["num_vertices"]
        except KeyError as exc:
            raise Exception(f"Record {name} contains multiple networks, please specify network name.") from exc

        # Retrieve CSV data
        url = f"{base_url}/net/{name}/files/{network}.csv.zip"
        try:
            response = request.urlopen(url)

            # decompress zip into temporary folder
            data = BytesIO(response.read())

            with zipfile.ZipFile(data, "r") as zip_ref:
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_ref.extractall(path=temp_dir)

                    # edges.csv contains edge list with edge properties
                    edges = pd.read_csv(
                        f"{temp_dir}/edges.csv", header=0, sep=",", skip_blank_lines=True, skipinitialspace=True
                    )

                    # rename columns
                    edges.rename(columns={"# source": "v", "target": "w"}, inplace=True)
                    if timestamps:
                        edges.rename(columns={time_attr: "t"}, inplace=True)

                    # construct graph and assign edge attributes
                    if not timestamps:
                        g = df_to_graph(
                            df=edges, multiedges=multiedges, is_undirected=not is_directed, num_nodes=num_nodes
                        )
                    else:
                        g = df_to_temporal_graph(df=edges, multiedges=multiedges, num_nodes=num_nodes)

                    # nodes.csv contains node indices with node properties (like name)
                    node_attrs = pd.read_csv(
                        f"{temp_dir}/nodes.csv", header=0, sep=",", skip_blank_lines=True, skipinitialspace=True
                    )
                    node_attrs.rename(columns={"# index": "index"}, inplace=True)

                    add_node_attributes(node_attrs, g)

                    # add graph-level attributes
                    for x in analyses:
                        g.data["analyses_" + x] = analyses[x]

                    return g
        except HTTPError as exc:
            msg = f"Could not retrieve netzschleuder record at {url}"
            raise Exception(msg) from exc
    except HTTPError as exc:
        msg = f"Could not retrieve netzschleuder record at {base_url}/api/net/{name}"
        raise Exception(msg) from exc
    return None
