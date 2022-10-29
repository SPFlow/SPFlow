# -*- coding: utf-8 -*-
"""Contains conditional Uniform leaf node for SPFlow in the ``torch`` backend.
"""
from typing import List, Union, Optional, Iterable, Tuple
from functools import reduce
import numpy as np
import torch
import torch.distributions as D

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.scope.scope import Scope
from spflow.torch.structure.module import Module
from spflow.torch.structure.nodes.leaves.parametric.uniform import Uniform
from spflow.base.structure.layers.leaves.parametric.uniform import (
    UniformLayer as BaseUniformLayer,
)


class UniformLayer(Module):
    r"""Layer of multiple (univariate) continuous Uniform distribution leaf nodes in the ``base`` backend.

    Represents multiple univariate Poisson distributions with independent scopes, each with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\text{end} - \text{start}}\mathbf{1}_{[\text{start}, \text{end}]}(x)

    where
        - :math:`x` is the input observation
        - :math:`\mathbf{1}_{[\text{start}, \text{end}]}` is the indicator function for the given interval (evaluating to 0 if x is not in the interval)

    Attributes:
        start:
            One-dimensional PyTorch tensor containing the start of the intervals (including).
        end:
            One-dimensional PyTorch tensor containing the end of the intervals (including). Must be larger than 'start'.
        end_next:
            One-dimensional PyTorch tensor containing the next largest floating point values to ``end``.
            Used for the PyTorch distributions which do not include the specified ends of the intervals.
        support_outside:
            One-dimensional PyTorch tensor containing booleans indicating whether or not values outside of the intervals are part of the support.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        start: Union[int, float, List[float], np.ndarray, torch.Tensor],
        end: Union[int, float, List[float], np.ndarray, torch.Tensor],
        support_outside: Union[
            bool, List[bool], np.ndarray, torch.Tensor
        ] = True,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``UniformLayer`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            start:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor containing the start of the intervals (including).
                If a single value is given, it is broadcast to all nodes.
            end:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor containing the end of the intervals (including). Must be larger than 'start'.
                If a single value is given, it is broadcast to all nodes.
            support_outside:
                Boolean, list of booleans or one-dimensional NumPy array or PyTorch tensor containing booleans indicating whether or not values outside of the intervals are part of the support.
                If a single boolean value is given, it is broadcast to all nodes.
                Defaults to True.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'UniformLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'UniformLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")

        super(UniformLayer, self).__init__(children=[], **kwargs)

        # register interval bounds as torch buffers (should not be changed)
        self.register_buffer("start", torch.empty(size=[]))
        self.register_buffer("end", torch.empty(size=[]))
        self.register_buffer("end_next", torch.empty(size=[]))
        self.register_buffer("support_outside", torch.empty(size=[]))

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(
            lambda s1, s2: s1.union(s2), self.scopes_out
        )

        # parse weights
        self.set_params(start, end, support_outside)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    def dist(self, node_ids: Optional[List[int]] = None) -> D.Distribution:
        r"""Returns the PyTorch distributions represented by the leaf layer.

        Args:
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            ``torch.distributions.Uniform`` instances.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        # create Torch distribution with specified parameters
        return D.Uniform(low=self.start[node_ids], high=self.end_next[node_ids])

    def set_params(
        self,
        start: Union[int, float, List[float], np.ndarray, torch.Tensor],
        end: Union[int, float, List[float], np.ndarray, torch.Tensor],
        support_outside: Union[bool, List[bool], np.ndarray, torch.Tensor],
    ) -> None:

        if isinstance(start, int) or isinstance(start, float):
            start = torch.tensor([start for _ in range(self.n_out)])
        elif isinstance(start, list) or isinstance(start, np.ndarray):
            start = torch.tensor(start)
        if start.ndim != 1:
            raise ValueError(
                f"Numpy array of 'start' values for 'UniformLayer' is expected to be one-dimensional, but is {start.ndim}-dimensional."
            )
        if start.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'start' values for 'UniformLayer' must match number of output nodes {self.n_out}, but is {start.shape[0]}"
            )

        if not torch.any(torch.isfinite(start)):
            raise ValueError(
                f"Values of 'start' for 'UniformLayer' must be finite, but was: {start}"
            )

        if isinstance(end, int) or isinstance(end, float):
            end = torch.tensor([end for _ in range(self.n_out)])
        elif isinstance(end, list) or isinstance(end, np.ndarray):
            end = torch.tensor(end)
        if end.ndim != 1:
            raise ValueError(
                f"Numpy array of 'end' values for 'UniformLayer' is expected to be one-dimensional, but is {end.ndim}-dimensional."
            )
        if end.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'end' values for 'UniformLayer' must match number of output nodes {self.n_out}, but is {end.shape[0]}"
            )

        if not torch.any(torch.isfinite(end)):
            raise ValueError(
                f"Value of 'end' for 'UniformLayer' must be finite, but was: {end}"
            )

        if not torch.all(start < end):
            raise ValueError(
                f"Lower bounds for Uniform distribution must be less than upper bounds, but were: {start}, {end}"
            )

        if isinstance(support_outside, bool):
            support_outside = torch.tensor(
                [support_outside for _ in range(self.n_out)]
            )
        elif isinstance(support_outside, list) or isinstance(
            support_outside, np.ndarray
        ):
            support_outside = torch.tensor(support_outside)
        if support_outside.ndim != 1:
            raise ValueError(
                f"Numpy array of 'support_outside' values for 'UniformLayer' is expected to be one-dimensional, but is {support_outside.ndim}-dimensional."
            )
        if support_outside.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'support_outside' values for 'UniformLayer' must match number of output nodes {self.n_out}, but is {support_outside.shape[0]}"
            )

        if not torch.any(torch.isfinite(support_outside)):
            raise ValueError(
                f"Value of 'support_outside' for 'UniformLayer' must be greater than 0, but was: {support_outside}"
            )

        # since torch Uniform distribution excludes the upper bound, compute next largest number
        end_next = torch.nextafter(end, torch.tensor(float("inf")))

        self.start.data = start
        self.end.data = end
        self.end_next.data = end_next
        self.support_outside.data = support_outside

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of three one-dimensional PyTorch tensor representing the starts and ends of the intervals and the booleans indicating whether or not values outside of the intervals are part of the supports.
        """
        return (self.start, self.end, self.support_outside)

    def check_support(
        self, data: torch.Tensor, node_ids: Optional[List[int]] = None, is_scope_data: bool=False
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Uniform distributions, which are:

        .. math::

            \text{supp}(\text{Uniform})=\begin{cases} [start,end] & \text{if support\_outside}=\text{false}\\
                                                 (-\infty,\infty) & \text{if support\_outside}=\text{true} \end{cases}
        where
            - :math:`start` is the start of the interval
            - :math:`end` is the end of the interval
            - :math:`\text{support\_outside}` is a truth value indicating whether values outside of the interval are part of the support

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
                Assumes that relevant data is located in the columns corresponding to the scope indices.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leafs' scope in the correct order (True) or if it needs to be extracted from the full data set.
                Note, that this should already only contain only the data according (and in order of) ``node_ids``.
                Defaults to False.

        Returns:
            Two dimensional PyTorch tensor indicating for each instance and node, whether they are part of the support (True) or not (False).
            Each row corresponds to an input sample.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        if is_scope_data:
            scope_data = data
        else:
            # all query scopes are univariate
            scope_data = data[
                :, [self.scopes_out[node_id].query[0] for node_id in node_ids]
            ]

        # torch distribution support is an interval, despite representing a distribution over a half-open interval
        # end is adjusted to the next largest number to make sure that desired end is part of the distribution interval
        # may cause issues with the support check; easier to do a manual check instead
        valid = torch.ones(scope_data.shape, dtype=torch.bool)

        # check if values are within valid range
        valid &= (scope_data >= self.start[torch.tensor(node_ids)]) & (
            scope_data < self.end[torch.tensor(node_ids)]
        )
        valid |= self.support_outside[torch.tensor(node_ids)]

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~(scope_data[~nan_mask & valid].isinf())

        return valid


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: UniformLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[UniformLayer, Uniform, None]:
    """Structural marginalization for ``UniformLayer`` objects in the ``torch`` backend.

    Structurally marginalizes the specified layer module.
    If the layer's scope contains non of the random variables to marginalize, then the layer is returned unaltered.
    If the layer's scope is fully marginalized over, then None is returned.

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
        Unaltered leaf layer or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    marginalized_node_ids = []
    marginalized_scopes = []

    for i, scope in enumerate(layer.scopes_out):

        # compute marginalized query scope
        marg_scope = [rv for rv in scope.query if rv not in marg_rvs]

        # node not marginalized over
        if len(marg_scope) == 1:
            marginalized_node_ids.append(i)
            marginalized_scopes.append(scope)

    if len(marginalized_node_ids) == 0:
        return None
    elif len(marginalized_node_ids) == 1 and prune:
        node_id = marginalized_node_ids.pop()
        return Uniform(
            scope=marginalized_scopes[0],
            start=layer.start[node_id].item(),
            end=layer.end[node_id].item(),
            support_outside=layer.support_outside[node_id].item(),
        )
    else:
        return UniformLayer(
            scope=marginalized_scopes,
            start=layer.start[marginalized_node_ids].detach(),
            end=layer.end[marginalized_node_ids].detach(),
            support_outside=layer.support_outside[
                marginalized_node_ids
            ].detach(),
        )


@dispatch(memoize=True)  # type: ignore
def toTorch(
    layer: BaseUniformLayer, dispatch_ctx: Optional[DispatchContext] = None
) -> UniformLayer:
    """Conversion for ``UniformLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return UniformLayer(
        scope=layer.scopes_out,
        start=layer.start,
        end=layer.end,
        support_outside=layer.support_outside,
    )


@dispatch(memoize=True)  # type: ignore
def toBase(
    layer: UniformLayer, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseUniformLayer:
    """Conversion for ``UniformLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseUniformLayer(
        scope=layer.scopes_out,
        start=layer.start.numpy(),
        end=layer.end.numpy(),
        support_outside=layer.support_outside.numpy(),
    )
