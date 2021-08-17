"""
Created on August 01, 2021

@authors: Kevin Huy Nguyen

This file provides the structural marginalization of a SPN.
"""

from copy import deepcopy
from .structural_transformations import prune
from .validity_checks import _isvalid_spn
from .node import SumNode, LeafNode, Node
import numpy as np
from typing import Set, List, Optional


def marginalize(node: Node, keep: List[int]) -> Optional[Node]:
    """
    Marginalizes the rvs not listed in keep.

    Args:
        node:
            SPN root node.
        keep:
            Set of features to keep.

    Returns: Root Node of marginalized SPN.
    """
    keep_set: Set = set(keep)

    def marg_recursive(node: Node) -> Optional[Node]:
        """
        Recursively goes through all children of node and updates its/their scopes and children according to the
        features in keep.

        Args:
            node:
                Node which will be recursively updated.

        Returns: Updated node.
        """
        new_node_scope: Set = keep_set.intersection(set(node.scope))

        if len(new_node_scope) == 0:
            # we are summing out this node
            return None

        if isinstance(node, LeafNode):
            if len(node.scope) > 1:
                raise Exception("Leaf Node with |scope| > 1")

            return deepcopy(node)

        children: List[Node] = []
        for c in node.children:
            new_c: Optional[Node] = marg_recursive(c)
            if new_c is None:
                continue
            children.append(new_c)

        if isinstance(node, SumNode):
            sum_node: SumNode = node.__class__(
                children=children, scope=list(new_node_scope), weights=np.array(node.weights)
            )
            new_node: Node = sum_node
        else:
            new_node = node.__class__(children=children, scope=list(new_node_scope))
        return new_node

    new_node = marg_recursive(node)

    if new_node is None:
        return None

    pruned_new_node: Node = prune(new_node)
    _isvalid_spn(pruned_new_node)

    return pruned_new_node
