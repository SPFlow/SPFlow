# -*- coding: utf-8 -*-
"""Contains conditional SPN-like sum layer for SPFlow in the ``torch`` backend.
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.data.scope import Scope
from spflow.torch.structure.module import Module
from spflow.base.structure.spn.layers.cond_sum_layer import (
    CondSumLayer as BaseCondSumLayer,
)

from typing import List, Union, Optional, Iterable, Callable
from copy import deepcopy
import numpy as np
import torch


class CondSumLayer(Module):
    r"""Layer representing multiple SPN-like sum nodes over all children in the ``torch`` backend.

    Represents multiple convex combinations of its children over the same scope.

    Methods:
        children():
            Iterator over all modules that are children to the module in a directed graph.

    Attributes:
        cond_f:
            Optional callable or list of callables to retrieve weights for the sum nodes.
            If a single callable, its output should be a dictionary containing ``weights`` as a key, and the value should be
            a list of floats, list of lists of floats, one- to two-dimensional NumPy array or PyTorch tensor,
            containing non-negative weights. There should be weights for each of the node and inputs.
            Each row corresponds to a sum node and values should sum up to one. If it is a list of floats
            or one-dimensional NumPy array, the same weights are reused for all sum nodes.
            If a list of callables, each one should return a dictionary containing ``weights`` as a key, and the value should
            be a list of floats or a one-dimensional NumPy array or PyTorch tensor containing non-zero values, summing up to one.
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
        cond_f: Optional[Union[Callable, List[Callable]]] = None,
        **kwargs,
    ) -> None:
        r"""Initializes ``CondSumLayer`` object.

        Args:
            n_nodes:
                Integer specifying the number of nodes the layer should represent.
            cond_f:
                Optional callable or list of callables to retrieve weights for the sum nodes.
                If a single callable, its output should be a dictionary containing ``weights`` as a key, and the value should be
                a list of floats, list of lists of floats, one- to two-dimensional NumPy array or PyTorch tensor,
                containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats
                or one-dimensional NumPy array, the same weights are reused for all sum nodes.
                If a list of callables, each one should return a dictionary containing ``weights`` as a key, and the value should
                be a list of floats or a one-dimensional NumPy array or PyTorch tensor containing non-zero values, summing up to one.

        Raises:
            ValueError: Invalid arguments.
        """
        if n_nodes < 1:
            raise ValueError(
                "Number of nodes for 'CondSumLayer' must be greater of equal to 1."
            )

        if not children:
            raise ValueError(
                "'CondSumLayer' requires at least one child to be specified."
            )

        super(CondSumLayer, self).__init__(children=children, **kwargs)

        self._n_out = n_nodes
        self.n_in = sum(child.n_out for child in self.children())

        # compute scope
        scope = None

        for child in children:
            for s in child.scopes_out:
                if scope is None:
                    scope = s
                else:
                    if not scope.equal_query(s):
                        raise ValueError(
                            f"'CondSumLayer' requires child scopes to have the same query variables."
                        )

                scope = scope.join(s)

        self.scope = scope

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
                If a single callable, its output should be a dictionary containing ``weights`` as a key, and the value should be
                a list of floats, list of lists of floats, one- to two-dimensional NumPy array or PyTorch tensor,
                containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats
                or one-dimensional NumPy array, the same weights are reused for all sum nodes.
                If a list of callables, each one should return a dictionary containing ``weights`` as a key, and the value should
                be a list of floats or a one-dimensional NumPy array or PyTorch tensor containing non-zero values, summing up to one.

        Raises:
            ValueError: If list of callables does not match number of nodes represented by the layer.
        """
        if isinstance(cond_f, List) and len(cond_f) != self.n_out:
            raise ValueError(
                "'CondSumLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes."
            )

        self.cond_f = cond_f

    def retrieve_params(
        self, data: torch.Tensor, dispatch_ctx: DispatchContext
    ) -> torch.Tensor:
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
            Two-dimensional PyTorch tensor of non-zero weights summing up to one per row.

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
                "'CondSumLayer' requires either 'weights' or 'cond_f' to retrieve 'weights' to be specified."
            )

        # if 'weights' was not already specified, retrieve it
        if weights is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, List):
                weights = torch.tensor([f(data)["weights"] for f in cond_f])
            else:
                weights = cond_f(data)["weights"]

        if isinstance(weights, list) or isinstance(weights, np.ndarray):
            weights = torch.tensor(weights).type(torch.get_default_dtype())
        if weights.ndim != 1 and weights.ndim != 2:
            raise ValueError(
                f"Torch tensor of weight values for 'CondSumLayer' is expected to be one- or two-dimensional, but is {weights.ndim}-dimensional."
            )
        if not torch.all(weights > 0):
            raise ValueError("Weights for 'SumLayer' must be all positive.")
        if not torch.allclose(weights.sum(dim=-1), torch.tensor(1.0)):
            raise ValueError(
                "Weights for 'CondSumLayer' must sum up to one in last dimension."
            )
        if not (weights.shape[-1] == self.n_in):
            raise ValueError(
                "Number of weights for 'CondSumLayer' in last dimension does not match total number of child outputs."
            )

        # same weights for all sum nodes
        if weights.ndim == 1:
            # broadcast weights to all nodes
            weights = torch.stack([weights for _ in range(self.n_out)])
        if weights.ndim == 2:
            # same weights for all sum nodes
            if weights.shape[0] == 1:
                # broadcast weights to all nodes
                weights = torch.concat(
                    [weights for _ in range(self.n_out)], dim=0
                )
            # different weights for all sum nodes
            elif weights.shape[0] == self.n_out:
                # already in correct output shape
                pass
            # incorrect number of specified weights
            else:
                raise ValueError(
                    f"Incorrect number of weights for 'CondSumLayer'. Size of first dimension must be either 1 or {self.n_out}, but is {weights.shape[0]}."
                )

        return weights


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: CondSumLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[None, CondSumLayer]:
    """Structural marginalization for conditional SPN-like sum layer objects in the ``torch`` backend.

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
        for child in layer.children():
            marg_child = marginalize(
                child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx
            )

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)

        return CondSumLayer(n_nodes=layer.n_out, children=marg_children)
    else:
        return deepcopy(layer)


@dispatch(memoize=True)  # type: ignore
def toBase(
    sum_layer: CondSumLayer, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseCondSumLayer:
    """Conversion for ``CondSumLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_layer:
            Layer to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondSumLayer(
        n_nodes=sum_layer.n_out,
        children=[
            toBase(child, dispatch_ctx=dispatch_ctx)
            for child in sum_layer.children()
        ],
    )


@dispatch(memoize=True)  # type: ignore
def toTorch(
    sum_layer: BaseCondSumLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> CondSumLayer:
    """Conversion for ``CondSumLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        sum_layer:
            Layer to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondSumLayer(
        n_nodes=sum_layer.n_out,
        children=[
            toTorch(child, dispatch_ctx=dispatch_ctx)
            for child in sum_layer.children
        ],
    )
