"""
Created on June 12, 2021

@authors: Kevin Huy Nguyen
"""
import numpy as np
from spn.base.nodes.node import (
    Node,
    LeafNode,
    ProductNode,
    SumNode,
    _print_node_graph,
    _get_leaf_nodes,
)
from typing import cast
from spn.base.rat.rat_spn import construct_spn
from spn.base.rat.region_graph import random_region_graph


def inference(spn: Node) -> float:
    """
    Runs inference on a given SPN. It calculates the values based on the structure of the SPN.

    :param spn: The SPN to run inference on. At least the values in the leaf node have to be populated.
    :return: A float value that represents the inferred value of the given SPN.
    """
    if np.isnan(spn.value):
        spn.value = calculate_node(spn)
    return spn.value


def calculate_node(node: Node) -> float:
    """
    Calculates the value of a node recursively by going through its children.
    The value of a SumNode is the sum of the values of its children weighted with their respective weights.
    This is done using the log-sum-exp trick to avoid numerical underflow.
    The value of a ProductNode is the product of the values of its children. Similarly, the values get summed up after
    being transformed into log-space to avoid numerical underflow.
    The value of a LeafNode is its own value, so its value has to be populated beforehand.

    :param node: A node to calculate its value from.
    :return: A float representing the value of the given node.
    """
    if type(node) is SumNode:
        node = cast(SumNode, node)
        # initialize value with -1 as this value is not possible in log space
        value: float = -1
        i = 0
        for child in node.children:
            if np.isnan(child.value):
                child_value: float = calculate_node(child)
            else:
                child_value = child.value
            if child_value != 0 and node.weights[i] != 0:
                if value == -1:
                    value = np.log(child_value) + np.log(node.weights[i])
                else:
                    value = np.logaddexp(value, np.log(child_value) + np.log(node.weights[i]))
            i += 1
        if value == -1:
            return 0
    elif type(node) is ProductNode:
        node = cast(ProductNode, node)
        value = 0
        for child in node.children:
            if child is LeafNode or not np.isnan(child.value):
                child_value = child.value
            else:
                child_value = calculate_node(child)
            if child_value != 0:
                value += np.log(child_value)
            else:
                return 0
    else:
        node = cast(LeafNode, node)
        if np.isnan(node.value):
            raise ValueError(
                "Please populate the values of all LeafNodes before running inference."
            )
        value = node.value
    return np.exp(value)


if __name__ == "__main__":
    region_graph = random_region_graph(X=set(range(1, 8)), depth=2, replicas=2)
    rat_spn = construct_spn(region_graph, 3, 2, 2)
    _print_node_graph(rat_spn.root_node)
    leaves = _get_leaf_nodes(rat_spn.root_node)
    # example population of leaves with value 0.5 to run inference
    i = 0
    for leaf in leaves:
        leaf.value = 0.5
        i += 1
    result = inference(rat_spn.root_node)
    print(result)
