"""
Created on May 05, 2021

@authors: Kevin Huy Nguyen, Bennet Wittelsbach

This file provides the basic components to build abstract probabilistic circuits, like SumNode, ProductNode, and LeafNode.
"""
from typing import List, Optional


class Node:
    """Base class for all types of nodes

    Attributes:
        children: A list of nodes containing the children of this node, or None.
        scope: A list of integers containing the scopes of this node, or None.
    """

    def __init__(self, children: List["Node"], scope: List[int]) -> None:
        # TODO: sollten Nodes auch IDs haben? (siehe SPFlow, z.B. fuer SPN-Ausgabe/Viz noetig)
        self.children = children
        self.scope = scope


class ProductNode(Node):
    """A ProductNode provides a factorization of its children, i.e. product nodes in SPNs have children with distinct scopes"""

    def __init__(self, children: List[Node], scope: List[int]) -> None:
        super().__init__(children=children, scope=scope)


class SumNode(Node):
    """A SumNode provides a weighted mixture of its children, i.e. sum nodes in SPNs have children with identical scopes

    Attributes:
        weights: A list of floats assigning a weight value to each of the SumNode's children.

    """

    def __init__(
        self, children: List[Node], scope: List[int], weights: List[float]
    ) -> None:
        super().__init__(children=children, scope=scope)
        self.weights = weights


class LeafNode(Node):
    """A LeafNode provides a probability distribution over some input variable(s)"""

    def __init__(self, children: Optional[List[Node]], scope: List[int]) -> None:
        # TODO: mit Steven abklaren, wie children in LeafNode behandelt werden; oder ob nicht Node, sondern SumNode/ProductNode children haben sollten
        if children is not None:
            raise ValueError("LeafNode must not have children")
        super().__init__(children=[], scope=scope)
