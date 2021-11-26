"""
Created on May 27, 2021

@authors: Philipp Deibert

This file provides the PyTorch variants of individual graph nodes.
"""
from multipledispatch import dispatch  # type: ignore
from typing import List, Union, Optional, Dict
import numpy as np

import torch
import torch.nn as nn

from spflow.torch.structure.module import TorchModule
from spflow.base.structure.nodes.node import INode, ISumNode, IProductNode, ILeafNode


def proj_convex_to_real(x: torch.Tensor) -> torch.Tensor:
    # convex coefficients are already normalized, so taking the log is sufficient
    return torch.log(x)


def proj_real_to_convex(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=-1)


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
        weights: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
    ) -> None:

        if not children:
            raise ValueError("Sum node must have at least one child.")

        if weights is None:
            weights = torch.rand(sum(len(child) for child in children)) + 1e-08  # avoid zeros
            weights /= weights.sum()

        self.num_in = sum(len(child) for child in children)

        super(TorchSumNode, self).__init__(children, scope)

        # register auxiliary parameters for weights as torch parameters
        self.weights_aux = nn.Parameter()

        self.weights = weights

    @property
    def weights(self) -> torch.Tensor:
        # project auxiliary weights onto weights that sum up to one
        return proj_real_to_convex(self.weights_aux)

    @weights.setter
    def weights(self, value: Union[np.ndarray, torch.Tensor, List[float]]) -> None:

        if isinstance(value, np.ndarray) or isinstance(value, list):
            value = torch.tensor(value)
        if not torch.all(value >= 0):
            raise ValueError("All weights must be non-negative.")
        if not torch.isclose(value.sum(), torch.tensor(1.0, dtype=value.dtype)):
            raise ValueError("Weights must sum up to one.")
        if not len(value) == self.num_in:
            raise ValueError("Number of weights does not match number of specified child nodes.")

        self.weights_aux.data = proj_convex_to_real(value)  # type: ignore

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # weight inputs in log-space
        weighted_inputs = inputs + self.weights.log()  # type: ignore
        return torch.logsumexp(weighted_inputs, dim=-1)


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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # return product (sum in log space)
        return torch.sum(inputs, dim=-1)


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

    def forward(self, inputs: torch.Tensor) -> None:
        pass


@dispatch(ILeafNode)  # type: ignore[no-redef]
def toTorch(x: ILeafNode) -> TorchLeafNode:
    return TorchLeafNode(scope=x.scope)


@dispatch(TorchLeafNode)  # type: ignore[no-redef]
def toNodes(x: TorchLeafNode) -> ILeafNode:
    return ILeafNode(scope=x.scope)
