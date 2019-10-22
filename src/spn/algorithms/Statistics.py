"""
Created on March 25, 2018

@author: Alejandro Molina
"""
from collections import Counter

from spn.structure.Base import get_nodes_by_type, Sum, Product, Leaf, get_number_of_edges, get_depth, Node
from spn.structure.leaves.parametric.Parametric import Parametric
import logging

logger = logging.getLogger(__name__)


def get_structure_stats_dict(node):
    nodes = get_nodes_by_type(node, Node)
    num_nodes = len(nodes)

    node_types = dict(Counter([type(n) for n in nodes]))

    edges = get_number_of_edges(node)
    layers = get_depth(node)

    params = 0
    for n in nodes:
        if isinstance(n, Sum):
            params += len(n.children)
        if isinstance(n, Leaf):
            params += len(n.parameters)

    result = {"nodes": num_nodes, "params": params, "edges": edges, "layers": layers, "count_per_type": node_types}
    return result


def get_structure_stats(node):
    num_nodes = len(get_nodes_by_type(node, Node))
    sum_nodes = get_nodes_by_type(node, Sum)
    n_sum_nodes = len(sum_nodes)
    n_prod_nodes = len(get_nodes_by_type(node, Product))
    leaf_nodes = get_nodes_by_type(node, Leaf)
    n_leaf_nodes = len(leaf_nodes)
    edges = get_number_of_edges(node)
    layers = get_depth(node)
    params = 0
    for n in sum_nodes:
        params += len(n.children)
    for l in leaf_nodes:
        params += len(l.parameters)


    return """---Structure Statistics---
# nodes             %s
    # sum nodes     %s
    # prod nodes    %s
    # leaf nodes    %s
# params            %s
# edges             %s
# layers            %s""" % (
        num_nodes,
        n_sum_nodes,
        n_prod_nodes,
        n_leaf_nodes,
        params,
        edges,
        layers,
    )
