# -*- coding: utf-8 -*-
"""Contains ``SumNode`` for SPFlow in the ``base`` backend.
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.base.structure.module import Module
from spflow.base.structure.general.nodes.node import Node

from typing import Optional, Union, Iterable, List
from copy import deepcopy
import numpy as np


class SumNode(Node):
    """SPN-like sum node in the ``base`` backend.

    Represents a convex combination of its children over the same scope.

    Attributes:
        children:
            Non-empty list of modules that are children to the node in a directed graph.
        weights:
            One-dimensional NumPy array containing non-negative weights for each input, summing up to one.
        n_out:
            Integer indicating the number of outputs. One for nodes.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(
        self,
        children: List[Module],
        weights: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> None:
        r"""Initializes ``SumNode`` object.

        Args:
            children:
                Non-empty list of modules that are children to the node.
                The output scopes for all child modules need to be equal.
            weights:
                Optional list of floats, or one-dimensional NumPy array containing non-negative weights for each input, summing up to one.
                Defaults to 'None' in which case weights are initialized to random weights in (0,1) and normalized.

        Raises:
            ValueError: Invalid arguments.
        """
        super(SumNode, self).__init__(children=children)

        if not children:
            raise ValueError(
                "'SumNode' requires at least one child to be specified."
            )

        scope = None

        for child in children:
            for s in child.scopes_out:
                if scope is None:
                    scope = s
                else:
                    if not scope.equal_query(s):
                        raise ValueError(
                            f"'SumNode' requires child scopes to have the same query variables."
                        )

                scope = scope.join(s)

        self.scope = scope
        self.n_in = sum(child.n_out for child in children)

        if weights is None:
            weights = np.random.rand(self.n_in) + 1e-08  # avoid zeros
            weights /= weights.sum()

        self.weights = weights

    @property
    def weights(self) -> np.ndarray:
        """Returns the weights of the node as a NumPy array."""
        return self._weights

    @weights.setter
    def weights(self, values: Union[np.ndarray, List[float]]) -> None:
        """Sets the weights of the node to specified values.

        Args:
            values:
                One-dimensional NumPy array or list of floats of non-negative values summing up to one.
                Number of values must match number of total inputs to the node.

        Raises:
            ValueError: Invalid values.
        """
        if isinstance(values, list):
            values = np.array(values)
        if values.ndim != 1:
            raise ValueError(
                f"Numpy array of weight values for 'SumNode' is expected to be one-dimensional, but is {values.ndim}-dimensional."
            )
        if not np.all(values > 0):
            raise ValueError("Weights for 'SumNode' must be all positive.")
        if not np.isclose(values.sum(), 1.0):
            raise ValueError("Weights for 'SumNode' must sum up to one.")
        if not (len(values) == self.n_in):
            raise ValueError(
                "Number of weights for 'SumNode' does not match total number of child outputs."
            )

        self._weights = values


@dispatch(memoize=True)  # type: ignore
def marginalize(
    sum_node: SumNode,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[SumNode, None]:
    r"""Structural marginalization for ``SumNode`` objects in the ``base`` backend.

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

        return SumNode(children=marg_children, weights=sum_node.weights)
    else:
        return deepcopy(sum_node)
