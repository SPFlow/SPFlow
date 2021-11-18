"""
Created on August 01, 2021

@authors: Kevin Huy Nguyen

This file provides algorithms for transforming the structure of a SPN.
"""

from .validity_checks import _isvalid_spn
from .node import ISumNode, IProductNode, ILeafNode, get_nodes_by_type, INode
import numpy as np
from typing import List, Type, cast


def prune(node: INode, contract_single_parents: bool = True) -> INode:
    """
    Goes through all nodes of SPN and prunes them. Unnecessary nodes such as child nodes of the same type as its parent
    are removed and its children correctly reconnected to the parent node. Weights of ISumNodes are recalculated.

    Args:
        node:
            Root node which will be pruned. Might be an invalid SPN as it is not pruned yet.
        contract_single_parents:
            Boolean indicating if a nodes child has only one child if the nodes child should be removed and its children
            then added to the node.

    Returns: Root node of pruned valid SPN.
    """
    nodes: List[INode] = get_nodes_by_type(node, (IProductNode, ISumNode))

    while len(nodes) > 0:
        n: INode = nodes.pop()
        n_type: Type = type(n)
        is_sum: bool = n_type == ISumNode

        i = 0
        while i < len(n.children):
            c: INode = n.children[i]

            if contract_single_parents and not isinstance(c, ILeafNode) and len(c.children) == 1:
                n.children[i] = c.children[0]
                continue

            if n_type == type(c):
                del n.children[i]
                n.children.extend(c.children)

                if is_sum:
                    c = cast(ISumNode, c)
                    n = cast(ISumNode, n)
                    w: np.ndarray = n.weights[i]
                    weights: List[float] = list(n.weights)
                    del weights[i]
                    weights.extend([cw * w for cw in list(c.weights)])
                    n.weights = np.array(weights)

                continue
            i += 1
        if is_sum and i > 0:
            n = cast(ISumNode, n)
            subtr: float = sum(list(n.weights)[1:])
            n.weights[0] = 1.0 - subtr

    if (
        contract_single_parents
        and isinstance(node, (IProductNode, ISumNode))
        and len(node.children) == 1
    ):
        node = node.children[0]

    _isvalid_spn(node)
    return node