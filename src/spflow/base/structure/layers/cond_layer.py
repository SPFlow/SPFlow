# -*- coding: utf-8 -*-
"""Contains basic conditional layer classes for SPFlow in the ``base`` backend.

Contains classes for layers of conditional SPN-like sum- and product nodes.
"""
from typing import List, Union, Optional, Iterable, Callable
from copy import deepcopy

import numpy as np

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.scope.scope import Scope
from spflow.base.structure.module import Module, NestedModule
from spflow.base.structure.nodes.cond_node import SPNCondSumNode


class SPNCondSumLayer(NestedModule):
    r"""Layer representing multiple SPN-like sum nodes over all children in the ``base`` backend.

    Represents multiple convex combinations of its children over the same scope.

    Attributes:
        children:
            Non-empty list of modules that are children to the node in a directed graph.
        cond_f:
            Optional callable or list of callables to retrieve weights for the sum nodes.
            If a single callable, its output should be a dictionary containing ``weights`` as a key, and the value should be
            a list of floats, list of lists of floats or one- to two-dimensional NumPy array,
            containing non-negative weights. There should be weights for each of the node and inputs.
            Each row corresponds to a sum node and values should sum up to one. If it is a list of floats
            or one-dimensional NumPy array, the same weights are reused for all sum nodes.
            If a list of callables, each one should return a dictionary containing ``weights`` as a key, and the value should
            be a list of floats or a one-dimensional NumPy array containing non-zero values, summing up to one.
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``SPNSumNode`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        n_nodes: int,
        children: List[Module],
        cond_f: Optional[Union[Callable, List[Callable]]] = None,
        **kwargs,
    ) -> None:
        r"""Initializes ``SPNCondSumLayer`` object.

        Args:
            n_nodes:
                Integer specifying the number of nodes the layer should represent.
            cond_f:
                Optional callable or list of callables to retrieve weights for the sum nodes.
                If a single callable, its output should be a dictionary containing 'weights' as a key, and the value should be
                a list of floats, list of lists of floats or one- to two-dimensional NumPy array,
                containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats
                or one-dimensional NumPy array, the same weights are reused for all sum nodes.
                If a list of callables, each one should return a dictionary containing 'weights' as a key, and the value should
                be a list of floats or a one-dimensional NumPy array containing non-zero values, summing up to one.

        Raises:
            ValueError: Invalid arguments.
        """
        if n_nodes < 1:
            raise ValueError(
                "Number of nodes for 'SPNCondSumLayer' must be greater of equal to 1."
            )

        if len(children) == 0:
            raise ValueError(
                "'SPNCondSumLayer' requires at least one child to be specified."
            )

        super(SPNCondSumLayer, self).__init__(children=children, **kwargs)

        self._n_out = n_nodes
        self.n_in = sum(child.n_out for child in self.children)

        # create input placeholder
        ph = self.create_placeholder(list(range(self.n_in)))

        # create sum nodes
        self.nodes = [SPNCondSumNode(children=[ph]) for _ in range(n_nodes)]

        # compute scope
        self.scope = self.nodes[0].scope

        self.set_cond_f(cond_f)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> List[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]

    def set_cond_f(
        self, cond_f: Optional[Union[List[Callable], Callable]] = None
    ) -> None:
        r"""Sets the ``cond_f`` property.

        Args:
            cond_f:
                Optional callable or list of callables to retrieve weights for the sum nodes.
                If a single callable, its output should be a dictionary containing 'weights' as a key, and the value should be
                a list of floats, list of lists of floats or one- to two-dimensional NumPy array,
                containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats
                or one-dimensional NumPy array, the same weights are reused for all sum nodes.
                If a list of callables, each one should return a dictionary containing 'weights' as a key, and the value should
                be a list of floats or a one-dimensional NumPy array containing non-zero values, summing up to one.

        Raises:
            ValueError: If list of callables does not match number of nodes represented by the layer.
        """
        if isinstance(cond_f, List) and len(cond_f) != self.n_out:
            raise ValueError(
                "'SPNCondSumLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes."
            )

        self.cond_f = cond_f

    def retrieve_params(
        self, data: np.ndarray, dispatch_ctx: DispatchContext
    ) -> np.ndarray:
        r"""Retrieves the conditional parameters of the leaf node.

        First, checks if conditional parameter (``weights``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function or list of functions (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameter.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameter.

        Args:
            data:
                Two-dimensional NumPy array containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            Two-dimensional NumPy array of non-zero weights summing up to one per row.

        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
        weights, cond_f = None, None

        # check dispatch cache for required conditional parameter 'weights'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if a value for 'weights' is specified (highest priority)
            if "weights" in args:
                weights = args["weights"]
            # check if alternative function to provide 'weights' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'weights' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'weights' nor 'cond_f' is specified (via node or arguments)
        if weights is None and cond_f is None:
            raise ValueError(
                "'SPNCondSumLayer' requires either 'weights' or 'cond_f' to retrieve 'weights' to be specified."
            )

        # if 'weights' was not already specified, retrieve it
        if weights is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, List):
                weights = np.array([f(data)["weights"] for f in cond_f])
            else:
                weights = cond_f(data)["weights"]

        if isinstance(weights, list):
            weights = np.array(weights)
        if weights.ndim != 1 and weights.ndim != 2:
            raise ValueError(
                f"Numpy array of weight values for 'SPNCondSumLayer' is expected to be one- or two-dimensional, but is {weights.ndim}-dimensional."
            )
        if not np.all(weights > 0):
            raise ValueError(
                "Weights for 'SPNCondSumLayer' must be all positive."
            )
        if not np.allclose(weights.sum(axis=-1), 1.0):
            raise ValueError(
                "Weights for 'SPNCondSumLayer' must sum up to one in last dimension."
            )
        if not (weights.shape[-1] == self.n_in):
            raise ValueError(
                "Number of weights for 'SPNCondSumLayer' in last dimension does not match total number of child outputs."
            )

        # same weights for all sum nodes
        if weights.ndim == 1:
            # broadcast weights to all nodes
            weights = np.stack([weights for _ in range(self.n_out)])
        if weights.ndim == 2:
            # same weights for all sum nodes
            if weights.shape[0] == 1:
                # broadcast weights to all nodes
                weights = np.concatenate(
                    [weights for _ in range(self.n_out)], axis=0
                )
            # different weights for all sum nodes
            elif weights.shape[0] == self.n_out:
                # already in correct output shape
                pass
            # incorrect number of specified weights
            else:
                raise ValueError(
                    f"Incorrect number of weights for 'SPNCondSumLayer'. Size of first dimension must be either 1 or {self.n_out}, but is {weights.shape[0]}."
                )

        return weights


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: SPNCondSumLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[SPNCondSumLayer, Module, None]:
    """Structural marginalization for conditional SPN-like sum layer objects in the ``base`` backend.

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
        for child in layer.children:
            marg_child = marginalize(
                child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx
            )

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)

        return SPNCondSumLayer(n_nodes=layer.n_out, children=marg_children)
    else:
        return deepcopy(layer)
