# -*- coding: utf-8 -*-
"""Contains conditional SPN-like sum node for SPFlow in the ``base`` backend.
"""
from abc import ABC
from typing import List, Union, Optional, Iterable, Callable
from copy import deepcopy

import numpy as np

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.data.scope import Scope
from spflow.base.structure.module import Module
from spflow.base.structure.nodes.node import Node


class SPNCondSumNode(Node):
    """Conditional SPN-like sum node in the ``base`` backend.

    Represents a convex combination of its children over the same scope.

    Attributes:
        children:
            Non-empty list of modules that are children to the node in a directed graph.
        cond_f:
            Optional callable to retrieve weights for the sum node.
            Its output should be a dictionary containing ``weights`` as a key, and the value should be
            a list of floats or a one-dimensional NumPy array containing non-zero values, summing up to one.
        n_out:
            Integer indicating the number of outputs. One for nodes.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(
        self, children: List[Module], cond_f: Optional[Callable] = None
    ) -> None:
        """Initializes ``SPNCondSumNode`` object.

        Args:
            children:
                Non-empty list of modules that are children to the node.
                The output scopes for all child modules need to be equal.
            cond_f:
                Optional callable to retrieve weights for the sum node.
                Its output should be a dictionary containing 'weights' as a key, and the value should be
                a list of floats or a one-dimensional NumPy array containing non-zero values, summing up to one.

        Raises:
            ValueError: Invalid arguments.
        """
        super(SPNCondSumNode, self).__init__(children=children)

        if not children:
            raise ValueError(
                "'SPNCondSumNode' requires at least one child to be specified."
            )

        scope = None

        for child in children:
            for s in child.scopes_out:
                if scope is None:
                    scope = s
                else:
                    if not scope.equal_query(s):
                        raise ValueError(
                            f"'SPNCondSumNode' requires child scopes to have the same query variables."
                        )

                scope = scope.join(s)

        self.scope = scope
        self.n_in = sum(child.n_out for child in children)

        self.cond_f = cond_f

    def set_cond_f(self, cond_f: Optional[Callable] = None) -> None:
        """Sets the function to retrieve the node's conditonal weights.

        Args:
            cond_f:
                Optional callable to retrieve weights for the sum node.
                Its output should be a dictionary containing ``weights`` as a key, and the value should be
                a list of floats or a one-dimensional NumPy array containing non-zero values, summing up to one.
        """
        self.cond_f = cond_f

    def retrieve_params(
        self, data: np.ndarray, dispatch_ctx: DispatchContext
    ) -> np.ndarray:
        """Retrieves the conditional weights of the sum node.

        First, checks if conditional weights (``weights``) are passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameters.

        Args:
            data:
                Two-dimensional NumPy array containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            One-dimensional NumPy array of non-zero weights

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
                "'SPNCondSumNode' requires either 'weights' or 'cond_f' to retrieve 'weights' to be specified."
            )

        # if 'weights' was not already specified, retrieve it
        if weights is None:
            weights = cond_f(data)["weights"]

        # check if value for 'weights' is valid
        if isinstance(weights, list):
            weights = np.array(weights)
        if weights.ndim != 1:
            raise ValueError(
                f"Numpy array of weight values for 'SPNCondSumNode' is expected to be one-dimensional, but is {weights.ndim}-dimensional."
            )
        if not np.all(weights > 0):
            raise ValueError(
                "Weights for 'CondSPNCondSumNode' must be all positive."
            )
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(
                "Weights for 'CondSPNCondSumNode' must sum up to one."
            )
        if not (len(weights) == self.n_in):
            raise ValueError(
                "Number of weights for 'CondSPNCondSumNode' does not match total number of child outputs."
            )

        return weights


@dispatch(memoize=True)  # type: ignore
def marginalize(
    sum_node: SPNCondSumNode,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
):
    """Structural marginalization for ``SPNCondSumNode`` objects in the ``base`` backend.

    Structurally marginalizes the specified sum node.
    If the sum node's scope contains non of the random variables to marginalize, then the node is returned unaltered.
    If the sum node's scope is fully marginalized over, then None is returned.
    If the sum node's scope is partially marginalized over, then a new sum node over the marginalized child modules is returned.

    Args:
        sum_node:
            Sum node module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect when marginalizing sum nodes. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        (Marginalized) sum node or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute node scope (node only has single output)
    node_scope = sum_node.scope

    mutual_rvs = set(node_scope.query).intersection(set(marg_rvs))

    # node scope is being fully marginalized
    if len(mutual_rvs) == len(node_scope.query):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:
        marg_children = []

        # marginalize child modules
        for child in sum_node.children:
            marg_child = marginalize(
                child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx
            )

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)

        return SPNCondSumNode(children=marg_children)
    else:
        return deepcopy(sum_node)
