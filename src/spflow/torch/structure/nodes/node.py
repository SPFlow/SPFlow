"""
Created on May 27, 2021

@authors: Philipp Deibert

This file provides the PyTorch variants of individual graph nodes.
"""
from multipledispatch import dispatch  # type: ignore
from typing import List
import numpy as np

import torch
from torch.nn.parameter import Parameter

from spflow.torch.structure.module import TorchModule
from spflow.python.structure.nodes.node import INode, ISumNode, IProductNode, ILeafNode


class TorchNode(TorchModule):
    """PyTorch version of an abstract node. See INode.

    Attributes:
        children (list(TorchNode)): List of child torch nodes (defaults to empty list).
        scope: List of integers containing the scopes of this node.
    """

    def __init__(self, children: List[TorchModule], scope: List[int]) -> None:

        super(TorchNode, self).__init__()

        # register children
        for i, child in enumerate(children):
            self.add_module("child_{}".format(i + 1), child)

        self.scope = scope

    def __len__(self) -> int:
        return 1


@dispatch(INode)  # type: ignore[no-redef]
def toTorch(x: INode) -> TorchNode:
    return TorchNode(children=[toTorch(child) for child in x.children], scope=x.scope)


@dispatch(TorchNode)  # type: ignore[no-redef]
def toNodes(x: TorchNode) -> INode:
    return INode(children=[toNodes(child) for child in x.children()], scope=x.scope)


class TorchSumNode(TorchNode):
    """PyTorch version of a sum node. See ISumNode.

    Attributes:
        children (list(TorchNode)): Non-empty list of child torch nodes.
        scope: Non-empty list of integers containing the scopes of this node.
        weights (list(float)): List of non-negative weights for each child (defaults to empty list and gets initialized
                               to random weights in [0,1)).
        normalize (bool): Boolean specifying whether or not to normalize the weights to sum up to one (defaults to True).
    """

    def __init__(
        self,
        children: List[TorchModule],
        scope: List[int],
        weights: np.ndarray = np.empty(0),
        normalize: bool = True,
    ) -> None:

        if not children:
            raise ValueError("Sum node must have at least one child.")

        # convert weight np.array to torch tensor
        # if no weights specified initialize weights randomly in [0,1)
        weights_torch: torch.Tensor = (
            torch.tensor(weights)
            if weights is not None
            else torch.rand(sum(len(child) for child in children))
        )

        if not torch.all(weights_torch >= 0):
            raise ValueError("All weights must be non-negative.")

        if not len(weights) == sum(len(child) for child in children):
            raise ValueError("Number of weights does not match number of specified child nodes.")

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


@dispatch(ISumNode)  # type: ignore[no-redef]
def toTorch(x: ISumNode) -> TorchSumNode:
    return TorchSumNode(
        children=[toTorch(child) for child in x.children],
        scope=x.scope,
        weights=x.weights,
    )


@dispatch(TorchSumNode)  # type: ignore[no-redef]
def toNodes(x: TorchSumNode) -> ISumNode:
    return ISumNode(children=[toNodes(child) for child in x.children()], scope=x.scope, weights=x.weights.detach().numpy())  # type: ignore[operator]


class TorchProductNode(TorchNode):
    """PyTorch version of a product node. See IProductNode.

    Attributes:
        children (list(TorchNode)): Non-empty list of child torch nodes.
        scope: Non-empty list of integers containing the scopes of this node.
    """

    def __init__(self, children: List[TorchModule], scope: List[int]) -> None:

        if not children:
            raise ValueError("Product node must have at least one child.")

        super(TorchProductNode, self).__init__(children, scope)

    def forward(self, x):
        # TODO: broadcast across batches
        # return weighted product
        return x.prod()


@dispatch(IProductNode)  # type: ignore[no-redef]
def toTorch(x: IProductNode) -> TorchProductNode:
    return TorchProductNode(children=[toTorch(child) for child in x.children], scope=x.scope)


@dispatch(TorchProductNode)  # type: ignore[no-redef]
def toNodes(x: TorchProductNode) -> IProductNode:
    return IProductNode(children=[toNodes(child) for child in x.children()], scope=x.scope)


class TorchLeafNode(TorchNode):
    def __init__(self, scope: List[int]) -> None:
        """PyTorch version of an abstract leaf node. See ILeafNode.

        Attributes:
            scope: Non-empty list of integers containing the scopes of this node.
        """
        super(TorchLeafNode, self).__init__([], scope)

    def forward(self, x):
        pass


@dispatch(ILeafNode)  # type: ignore[no-redef]
def toTorch(x: ILeafNode) -> TorchLeafNode:
    return TorchLeafNode(scope=x.scope)


@dispatch(TorchLeafNode)  # type: ignore[no-redef]
def toNodes(x: TorchLeafNode) -> ILeafNode:
    return ILeafNode(scope=x.scope)
