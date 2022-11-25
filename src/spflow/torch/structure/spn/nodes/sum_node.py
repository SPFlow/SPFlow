"""Contains ``SumNode`` for SPFlow in the ``torch`` backend.
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.data.scope import Scope
from spflow.base.structure.spn.nodes.sum_node import (
    SumNode as BaseSumNode,
)
from spflow.torch.structure.module import Module
from spflow.torch.structure.general.nodes.node import Node
from spflow.torch.utils.projections import (
    proj_convex_to_real,
    proj_real_to_convex,
)

from typing import List, Union, Optional, Iterable
from copy import deepcopy
import numpy as np
import torch


class SumNode(Node):
    """SPN-like sum node in the ``torch`` backend.

    Represents a convex combination of its children over the same scope.
    Internally, the weights are represented as unbounded parameters that are projected onto convex combination for representing the actual weights.

    Methods:
        children():
            Iterator over all modules that are children to the module in a directed graph.

    Attributes:
        weights_aux:
            One-dimensional PyTorch tensor containing weights for each input.
        weights:
            One-dimensional PyTorch tensor containing non-negative weights for each input, summing up to one (projected from 'weights_aux').
        n_out:
            Integer indicating the number of outputs. One for nodes.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(
        self,
        children: List[Module],
        weights: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
    ) -> None:
        """Initializes 'SumNode' object.

        Args:
            children:
                Non-empty list of modules that are children to the node.
                The output scopes for all child modules need to be equal.
            weights:
                Optional list of floats, one-dimensional NumPy array or one-dimensional PyTorch tensor containing non-negative weights for each input, summing up to one.
                Defaults to 'None' in which case weights are initialized to random weights in (0,1) and normalized.

        Raises:
            ValueError: Invalid arguments.
        """
        super().__init__(children=children)

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
            weights = torch.rand(self.n_in) + 1e-08  # avoid zeros
            weights /= weights.sum()

        # register auxiliary parameters for weights as torch parameters
        self.weights_aux = torch.nn.Parameter()
        # initialize weights
        self.weights = weights

    @property
    def weights(self) -> torch.Tensor:
        """Returns the weights of the node as a PyTorch tensor."""
        # project auxiliary weights onto weights that sum up to one
        return proj_real_to_convex(self.weights_aux)

    @weights.setter
    def weights(
        self, values: Union[np.ndarray, torch.Tensor, List[float]]
    ) -> None:
        """Sets the weights of the node to specified values.

        Args:
            values:
                One-dimensional NumPy array, PyTorch tensor or list of floats of non-negative values summing up to one.
                Number of values must match number of total inputs to the node.

        Raises:
            ValueError: Invalid values.
        """
        if isinstance(values, list) or isinstance(values, np.ndarray):
            values = torch.tensor(values).float()
        if values.ndim != 1:
            raise ValueError(
                f"Torch tensor of weight values for 'SumNode' is expected to be one-dimensional, but is {values.ndim}-dimensional."
            )
        if not torch.all(values > 0):
            raise ValueError("Weights for 'SumNode' must be all positive.")
        if not torch.isclose(
            values.sum(), torch.tensor(1.0, dtype=values.dtype)
        ):
            raise ValueError("Weights for 'SumNode' must sum up to one.")
        if not (len(values) == self.n_in):
            raise ValueError(
                "Number of weights for 'SumNode' does not match total number of child outputs."
            )

        self.weights_aux.data = proj_convex_to_real(values)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    sum_node: SumNode,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
):
    """Structural marginalization for ``SumNode`` objects in the ``torch`` backend.

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
        for child in sum_node.children():
            marg_child = marginalize(
                child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx
            )

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)

        return SumNode(children=marg_children, weights=sum_node.weights)
    else:
        return deepcopy(sum_node)


@dispatch(memoize=True)  # type: ignore
def toBase(
    sum_node: SumNode, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseSumNode:
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseSumNode(
        children=[
            toBase(child, dispatch_ctx=dispatch_ctx)
            for child in sum_node.children()
        ],
        weights=sum_node.weights.detach().cpu().numpy(),
    )


@dispatch(memoize=True)  # type: ignore
def toTorch(
    sum_node: BaseSumNode, dispatch_ctx: Optional[DispatchContext] = None
) -> SumNode:
    """Conversion for ``SumNode`` from ``base`` backend to ``torch`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return SumNode(
        children=[
            toTorch(child, dispatch_ctx=dispatch_ctx)
            for child in sum_node.children
        ],
        weights=sum_node.weights,
    )
