"""
Created on May 05, 2021

@authors: Kevin Huy Nguyen, Bennet Wittelsbach

This file provides the basic components to build abstract probabilistic circuits, like SumNode, ProductNode, and LeafNode.
"""
from typing import List, Optional
import math

class Node:
    """Base class for all types of nodes

    Attributes:
        children: A list of nodes containing the children of this node, or None.
        scope: A list of integers containing the scopes of this node, or None.
    """
    
    scope: List[int]

    def __init__(self, children: List["Node"], scope: List[int]) -> None:
        # TODO: sollten Nodes auch IDs haben? (siehe SPFlow, z.B. fuer SPN-Ausgabe/Viz noetig)
        self.children = children
        self.scope = scope

    def __str__(self) -> str:
        '''
        Ad-hoc method to print structure of node and children (for debugging purposes)
        '''
        strings = [self.__class__.__name__]
        strings += [str(child).replace("\n", "\n    ") for child in self.children]

        return "\n    ".join(strings)

    def __eq__(self, other) -> bool:
        '''
        Checks whether two objects are identical by comparing their class, scope and children (recursively). 
        '''
        return (type(self) is type(other)
            and self.scope == other.scope
            and self.children == other.children)

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
    
    def __eq__(self, other) -> bool:
        '''
        Checks whether two objects are identical by comparing their class, scope, children (recursively) and weights.
        Note that weight comparison is done approximately due to numerical issues when conversion between graph representations.
        '''
        return (super().__eq__(other)
            and all(map(lambda x,y: math.isclose(x,y,rel_tol=1.e-5), self.weights, other.weights))
            and len(self.weights) == len(other.weights))


class LeafNode(Node):
    """A LeafNode provides a probability distribution over some input variable(s)"""

    def __init__(self, scope: List[int]) -> None:
        # TODO: mit Steven abklaren, wie children in LeafNode behandelt werden; oder ob nicht Node, sondern SumNode/ProductNode children haben sollten
    
        super().__init__(children=[], scope=scope)
