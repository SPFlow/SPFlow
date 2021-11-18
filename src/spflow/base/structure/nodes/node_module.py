"""
Created on October 21, 2021

@authors: Kevin Huy Nguyen
"""

import numpy as np
from typing import List
from spflow.base.structure.module import Module
from spflow.base.structure.nodes.node import (
    ILeafNode,
    IProductNode,
    ISumNode,
    INode,
)
from spflow.base.structure.network_type import NetworkType


class Node(Module):
    """Base class for all types of nodes modules

    Attributes:
        output_nodes:
            List of one INode as Node modules only encapsulate a single INode, which is at the same time the root.
        nodes:
            List of one INode as Node modules only encapsulate a single INode
    """

    def __init__(self, scope: List[int], children: List[Module], network_type: NetworkType) -> None:
        super().__init__(children=children, network_type=network_type, scope=scope)
        self.output_nodes: List[INode] = []
        self.nodes: List[INode] = []

    def __len__(self):
        return 1


class SumNode(Node):
    """SumNode is module encapsulating one ISumNode.

    Args:
        weights:
            A np.array of floats assigning a weight value to each of the encapsulated ISumNode's children.
    """

    def __init__(
        self,
        scope: List[int],
        weights: np.ndarray,
        children: List[Module],
        network_type: NetworkType,
    ) -> None:
        super().__init__(children=children, network_type=network_type, scope=scope)

        # check if all children are Modules and all values in weights are floats
        assert all([issubclass(type(obj), Module) for obj in self.children])
        assert all([issubclass(type(obj), float) for obj in weights])

        # connect the output INodes of the child modules to this modules INode
        node_children = []
        for module in self.children:
            for root in module.output_nodes:
                node_children.append(root)

        # check if there the same amount of weights as there are children output nodes
        assert len(node_children) == len(weights)

        # check if weights sum to 1
        assert np.isclose(sum(weights), 1.0)

        node: INode = ISumNode(children=node_children, scope=scope, weights=weights)
        self.nodes: List[INode] = [node]
        self.output_nodes: List[INode] = [node]

    def __len__(self):
        return 1


class ProductNode(Node):
    """ProductNode is module encapsulating one IProductNode."""

    def __init__(self, scope: List[int], children: List[Module], network_type: NetworkType) -> None:
        super().__init__(children=children, network_type=network_type, scope=scope)

        # check if all children are Modules
        assert all([issubclass(type(obj), Module) for obj in self.children])

        # connect the output INodes of the child modules to this modules INode
        node_children = []
        for module in self.children:
            for root in module.output_nodes:
                node_children.append(root)

        node: INode = IProductNode(children=node_children, scope=scope)
        self.nodes: List[INode] = [node]
        self.output_nodes: List[INode] = [node]

    def __len__(self):
        return 1


class LeafNode(Node):
    """LeafNode is module encapsulating one ILeafNode.

    Args:
        children:
            Empty list as LeafNodes can not have children.
    """

    def __init__(
        self,
        scope: List[int],
        network_type: NetworkType,
    ) -> None:
        super().__init__(children=[], network_type=network_type, scope=scope)
        node: INode = ILeafNode(scope=scope)
        self.nodes: List[INode] = [node]
        self.output_nodes: List[INode] = [node]

    def __len__(self):
        return 1
