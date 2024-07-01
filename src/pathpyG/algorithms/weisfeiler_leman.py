from __future__ import annotations
from typing import Tuple, List, Dict

from pathpyG.core.Graph import Graph

def WeisfeilerLeman_test(g1: Graph, g2: Graph, features_g1: dict = None, features_g2: dict = None) -> Tuple[bool, List[str], List[str]]:
    """Run Weisfeiler-Leman isomorphism test on two graphs.
    
    The algorithm heuristically checks whether two graphs are isomorphic. If it returns False,
    we can be sure that the graphs are non-isomoprhic. If the test returns True we did not find
    conclusive evidence that they are not isomorphic, i.e. the graphs may or may not be isomophic.

    The two graphs must have IndexMap mappings that assign different node IDs to the nodes
    in both graphs. The function will raise an error if the node labels of both graphs overlap.

    The function returns a tuple (bool, list, list), where the first entry is the result of the test
    and the two lists represent the fingerprints of the two graphs. If the test yields true the fingerprints
    are identical. If the test fails, the fingerprints do not correspond.

    Args:
        g1: pp.Graph
        g2: pp.Graph
    """
    if g1.mapping is None or g2.mapping is None:
        raise Exception('Graphs must contain IndexMap that assigns node IDs')
    if len(set(g1.mapping.node_ids).intersection(g2.mapping.node_ids)) > 0:
        raise Exception('node identifiers of graphs must not overlap')
    g_combined = g1 + g2
    # initialize labels of all nodes to zero
    if features_g1 is None or features_g2 is None:       
        fingerprint: Dict[str | int, str] = {v: '0' for v in g_combined.nodes}
    else:
        fingerprint = features_g1.copy()
        fingerprint.update(features_g2)
    labels = {} 
    label_count = 1
    stop = False
    while not stop:
        new_fingerprint = {} 
        for node in g_combined.nodes:
            # create new label based on own label and sorted labels of all neighbors
            n_label = [fingerprint[x] for x in g_combined.successors(node)]
            n_label.sort()
            label = str(fingerprint[node]) + str(n_label)
            # previously unknown label
            if label not in labels:
                # create a new label based on next consecutive number
                labels[label] = label_count
                label_count += 1 
            new_fingerprint[node] = labels[label]        
        if len(set(fingerprint.values())) == len(set(new_fingerprint.values())):
            # we processed all nodes in both graphs without encountering a new label, so we stop
            stop = True
        else:
            # update fingerprint and continue
            fingerprint = new_fingerprint.copy()

    # Reduce fingerprints to nodes of g1 and g2 respectively
    fingerprint_1 = [fingerprint[v] for v in g1.nodes]
    fingerprint_1_sorted = fingerprint_1.copy()
    fingerprint_1_sorted.sort()
    fingerprint_2 = [fingerprint[v] for v in g2.nodes]
    fingerprint_2_sorted = fingerprint_2.copy()
    fingerprint_2_sorted.sort()
    
    # perform WL-test
    if fingerprint_1_sorted == fingerprint_2_sorted:
        return True, fingerprint_1, fingerprint_2
    return False, fingerprint_1, fingerprint_2
