"""Contains SPN-like sum layer for SPFlow in the ``base`` backend.
"""
from copy import deepcopy
from typing import Iterable, List, Optional, Union

import numpy as np

from spflow.base.structure.module import Module
from spflow.base.structure.nested_module import NestedModule
from spflow.base.structure.spn.nodes.sum_node import SumNode
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class SumLayer(NestedModule):
    r"""Layer representing multiple SPN-like sum nodes over all children in the ``base`` backend.

    Represents multiple convex combinations of its children over the same scope.

    Attributes:
        children:
            Non-empty list of modules that are children to the node in a directed graph.
        weights:
            Two-dimensional NumPy array containing non-negative weights for each input.
            Each row corresponds to a sum node with values summing up to one.
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``SumNode`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        n_nodes: int,
        children: List[Module],
        weights: Optional[Union[np.ndarray, List[List[float]], List[float]]] = None,
        **kwargs,
    ) -> None:
        r"""Initializes ``SumLayer`` object.

        Args:
            n_nodes:
                Integer specifying the number of nodes the layer should represent.
            children:
                Non-empty list of modules that are children to the layer.
                The output scopes for all child modules need to be equal.
            weights:
                Optional list of floats, list of lists of floats or one- to two-dimensional NumPy array,
                containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats
                or one-dimensional NumPy array, the same weights are reused for all sum nodes.
                Defaults to 'None' in which case weights are initialized to random weights in (0,1) and normalized per row.

        Raises:
            ValueError: Invalid arguments.
        """
        if n_nodes < 1:
            raise ValueError("Number of nodes for 'SumLayer' must be greater of equal to 1.")

        if len(children) == 0:
            raise ValueError("'SumLayer' requires at least one child to be specified.")

        super().__init__(children=children, **kwargs)

        self._n_out = n_nodes
        self.n_in = sum(child.n_out for child in self.chs)

        # create input placeholder
        ph = self.create_placeholder(list(range(self.n_in)))

        # create sum nodes
        self.nodes = [SumNode(children=[ph]) for _ in range(n_nodes)]

        # parse weights
        if weights is not None:
            self.weights = weights

        # compute scope
        self.scope = self.nodes[0].scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> List[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]

    @property
    def weights(self) -> np.ndarray:
        """Returns the weights of all nodes as a two-dimensional NumPy array."""
        return np.vstack([node.weights for node in self.nodes])

    @weights.setter
    def weights(self, values: Union[np.ndarray, List[List[float]], List[float]]) -> None:
        """Sets the weights of all nodes to specified values.

        Args:
            values:
                List of floats, list of lists of floats or one- to two-dimensional NumPy array,
                containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats
                or one-dimensional NumPy array, the same weights are reused for all sum nodes.
                Two-dimensional NumPy array containing non-negative weights for each input.

        Raises:
            ValueError: Invalid values.
        """
        if isinstance(values, list):
            values = np.array(values)
        if values.ndim != 1 and values.ndim != 2:
            raise ValueError(
                f"Numpy array of weight values for 'SumLayer' is expected to be one- or two-dimensional, but is {values.ndim}-dimensional."
            )
        if not np.all(values > 0):
            raise ValueError("Weights for 'SumLayer' must be all positive.")
        if not np.allclose(values.sum(axis=-1), 1.0):
            raise ValueError("Weights for 'SumLayer' must sum up to one in last dimension.")
        if not (values.shape[-1] == self.n_in):
            raise ValueError(
                "Number of weights for 'SumLayer' in last dimension does not match total number of child outputs."
            )

        # same weights for all sum nodes
        if values.ndim == 1:
            for node in self.nodes:
                node.weights = values.copy()
        if values.ndim == 2:
            # same weights for all sum nodes
            if values.shape[0] == 1:
                for node in self.nodes:
                    node.weights = values.squeeze(0).copy()
            # different weights for all sum nodes
            elif values.shape[0] == self.n_out:
                for node, node_values in zip(self.nodes, values):
                    node.weights = node_values.copy()
            # incorrect number of specified weights
            else:
                raise ValueError(
                    f"Incorrect number of weights for 'SumLayer'. Size of first dimension must be either 1 or {self.n_out}, but is {values.shape[0]}."
                )


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: SumLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[SumLayer, Module, None]:
    """Structural marginalization for SPN-like sum layer objects in the ``base`` backend.

    Structurally marginalizes the specified layer module.
    If the layer's scope contains non of the random variables to marginalize, then the layer is returned unaltered.
    If the layer's scope is fully marginalized over, then None is returned.
    If the layer's scope is partially marginalized over, then a new sum layer over the marginalized child modules is returned.

    Args:
        layer:
            Layer module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect here. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        (Marginalized) sum layer or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute node scope (node only has single output)
    layer_scope = layer.scope

    mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

    # node scope is being fully marginalized
    if len(mutual_rvs) == len(layer_scope.query):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:
        # TODO: pruning
        marg_children = []

        # marginalize child modules
        for child in layer.chs:
            marg_child = marginalize(child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)

        return SumLayer(n_nodes=layer.n_out, children=marg_children, weights=layer.weights)
    else:
        return deepcopy(layer)
