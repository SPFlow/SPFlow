"""
Created on May 27, 2021

@authors: Philipp Deibert

This file provides the PyTorch variants of individual graph nodes.
"""
from multimethod import multimethod
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from spn.structure.graph.Node import Node, SumNode, ProductNode, LeafNode


class TorchNode(nn.Module):
    """PyTorch version of an abstract node. See Node.

    Attributes:
        children (list(TorchNode)): List of child torch nodes (defaults to empty list).
        scope: List of integers containing the scopes of this node.
    """

    scope: List[int]

    def __init__(self, children: List["TorchNode"], scope: List[int]) -> None:

        super(TorchNode, self).__init__()

        # register children
        for i, child in enumerate(children):
            self.add_module("child_{}".format(i + 1), child)

        self.scope = scope


@multimethod  # type: ignore[no-redef]
def toTorch(x: Node) -> TorchNode:
    return TorchNode(children=[toTorch(child) for child in x.children], scope=x.scope)


@multimethod  # type: ignore[no-redef]
def toNodes(x: TorchNode) -> Node:
    return Node(children=[toNodes(child) for child in x.children()], scope=x.scope)


class TorchSumNode(TorchNode):
    """PyTorch version of a sum node. See SumNode.

    Attributes:
        children (list(TorchNode)): Non-empty list of child torch nodes.
        scope: Non-empty list of integers containing the scopes of this node.
        weights (list(float)): List of non-negative weights for each child (defaults to empty list and gets initialized to random weights in [0,1)).
        normalize (bool): Boolean specifying whether or not to normalize the weights to sum up to one (defaults to True).
    """

    def __init__(
        self,
        children: List[TorchNode],
        scope: List[int],
        weights: Optional[List[float]] = [],
        normalize: bool = True,
    ) -> None:

        assert len(children) > 0, "Sum node must have at least one child."

        # convert weight list to torch tensor
        # if no weights specified initialize weights randomly in [0,1)
        weights_torch: torch.Tensor = (
            torch.tensor(weights) if weights else torch.rand(len(children))
        )

        assert torch.all(weights_torch >= 0), "All weights must be non-negative."

        # noramlize
        if normalize:
            weights_torch /= weights_torch.sum()

        super(TorchSumNode, self).__init__(children, scope)

        # store weight parameters
        self.register_parameter("weights", Parameter(weights_torch))

    def forward(self, x):
        # TODO: broadcast across batches
        # return weighted sum
        return (x * self.weights).sum()


@multimethod  # type: ignore[no-redef]
def toTorch(x: SumNode) -> TorchSumNode:
    return TorchSumNode(
        children=[toTorch(child) for child in x.children],
        scope=x.scope,
        weights=x.weights,
    )


@multimethod  # type: ignore[no-redef]
def toNodes(x: TorchSumNode) -> SumNode:
    return SumNode(children=[toNodes(child) for child in x.children()], scope=x.scope, weights=x.weights.tolist())  # type: ignore[operator]


class TorchProductNode(TorchNode):
    """PyTorch version of a product node. See ProductNode.

    Attributes:
        children (list(TorchNode)): Non-empty list of child torch nodes.
        scope: Non-empty list of integers containing the scopes of this node.
    """

    def __init__(self, children: List[TorchNode], scope: List[int]) -> None:

        assert len(children) > 0, "Sum node must have at least one child."

        super(TorchProductNode, self).__init__(children, scope)

    def forward(self, x):
        # TODO: broadcast across batches
        # return weighted product
        return x.prod()


@multimethod  # type: ignore[no-redef]
def toTorch(x: ProductNode) -> TorchProductNode:
    return TorchProductNode(
        children=[toTorch(child) for child in x.children], scope=x.scope
    )


@multimethod  # type: ignore[no-redef]
def toNodes(x: TorchProductNode) -> ProductNode:
    return ProductNode(
        children=[toNodes(child) for child in x.children()], scope=x.scope
    )


class TorchLeafNode(TorchNode):
    def __init__(self, scope: List[int]) -> None:
        """PyTorch version of an abstract leaf node. See LeafNode.

        Attributes:
            scope: Non-empty list of integers containing the scopes of this node.
        """
        super(TorchLeafNode, self).__init__([], scope)

    def forward(self, x):
        pass


@multimethod  # type: ignore[no-redef]
def toTorch(x: LeafNode) -> TorchLeafNode:
    return TorchLeafNode(scope=x.scope)


@multimethod  # type: ignore[no-redef]
def toNodes(x: TorchLeafNode) -> LeafNode:
    return LeafNode(scope=x.scope)
