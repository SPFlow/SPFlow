# -*- coding: utf-8 -*-
"""Contains Gaussian leaf layer for SPFlow in the ``torch`` backend.
"""
from typing import List, Union, Optional, Iterable, Tuple
from functools import reduce
import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from ....nodes.leaves.parametric.projections import (
    proj_bounded_to_real,
    proj_real_to_bounded,
)

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.data.scope import Scope
from spflow.torch.structure.module import Module
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.layers.leaves.parametric.gaussian import (
    GaussianLayer as BaseGaussianLayer,
)


class GaussianLayer(Module):
    r"""Layer of multiple (univariate) Gaussian distribution leaf nodes in the ``torch`` backend.

    Represents multiple univariate Gaussian distributions with independent scopes, each with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})

    where
        - :math:`x` the observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Internally :math:`\mu,\sigma` are represented as unbounded parameters that are projected onto the bounded range :math:`(0,\infty)` for representing the actual means and standard deviations, respectively.

    Attributes:
        mean:
            One-dimensional PyTorch tensor representing the mean (:math:`\mu`) of the Gamma distribution.
        std_aux:
            Unbounded one-dimensional PyTorch parameter that is projected to yield the actual standard deviations.
        std:
            One-dimensional PyTorch tensor representing the standard deviations (:math:`\sigma`) of the Gaussian distributions, greater than 0 (projected from ``std_aux``).
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        mean: Union[int, float, List[float], np.ndarray, torch.Tensor] = 0.0,
        std: Union[int, float, List[float], np.ndarray, torch.Tensor] = 1.0,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``GaussianLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            mean:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor representing the means (:math:`\mu`).
                If a single value is given it is broadcast to all nodes.
                Defaults to 0.0.
            std:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor representing the standard deviations (:math:`\sigma`), greater than 0.
                If a single value is given it is broadcast to all nodes.
                Defaults to 1.0.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'GaussianLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError(
                    "List of scopes for 'GaussianLayer' was empty."
                )

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")

        super(GaussianLayer, self).__init__(children=[], **kwargs)

        # register auxiliary torch parameter for rate l of each implicit node
        self.mean = Parameter()
        self.std_aux = Parameter()

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(
            lambda s1, s2: s1.union(s2), self.scopes_out
        )

        # parse weights
        self.set_params(mean, std)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def std(self) -> torch.Tensor:
        """TODO"""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.std_aux, lb=0.0)  # type: ignore

    def dist(self, node_ids: Optional[List[int]] = None) -> D.Distribution:
        r"""Returns the PyTorch distributions represented by the leaf layer.

        Args:
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            ``torch.distributions.Normal`` instance.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Normal(loc=self.mean[node_ids], scale=self.std[node_ids])

    def set_params(
        self,
        mean: Union[int, float, List[float], np.ndarray, torch.Tensor],
        std: Union[int, float, List[float], np.ndarray, torch.Tensor],
    ) -> None:
        r"""Sets the parameters for the represented distributions.

        TODO: projection function

        Args:
            mean:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor representing the means (:math:`\mu`).
                If a single value is given it is broadcast to all nodes.
                Defaults to 0.0.
            std:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor representing the standard deviations (:math:`\sigma`), greater than 0.
                If a single value is given it is broadcast to all nodes.
                Defaults to 1.0.
        """
        if isinstance(mean, int) or isinstance(mean, float):
            mean = torch.tensor([mean for _ in range(self.n_out)])
        elif isinstance(mean, list) or isinstance(mean, np.ndarray):
            mean = torch.tensor(mean)
        if mean.ndim != 1:
            raise ValueError(
                f"Numpy array of 'mean' values for 'GaussianLayer' is expected to be one-dimensional, but is {mean.ndim}-dimensional."
            )
        if mean.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'mean' values for 'GaussianLayer' must match number of output nodes {self.n_out}, but is {mean.shape[0]}"
            )

        if not torch.any(torch.isfinite(mean)):
            raise ValueError(
                f"Values of 'mean' for 'GaussianLayer' must be finite, but was: {mean}"
            )

        if isinstance(std, int) or isinstance(std, float):
            std = torch.tensor([std for _ in range(self.n_out)])
        elif isinstance(std, list) or isinstance(std, np.ndarray):
            std = torch.tensor(std)
        if std.ndim != 1:
            raise ValueError(
                f"Numpy array of 'std' values for 'GaussianLayer' is expected to be one-dimensional, but is {std.ndim}-dimensional."
            )
        if std.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'std' values for 'GaussianLayer' must match number of output nodes {self.n_out}, but is {std.shape[0]}"
            )

        if torch.any(std <= 0.0) or not torch.any(torch.isfinite(std)):
            raise ValueError(
                f"Value of 'std' for 'GaussianLayer' must be greater than 0, but was: {std}"
            )

        self.mean.data = mean
        self.std_aux.data = proj_bounded_to_real(std, lb=0.0)

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of one-dimensional PyTorch tensors representing the means and standard deviations.
        """
        return (self.mean, self.std)

    def check_support(
        self,
        data: torch.Tensor,
        node_ids: Optional[List[int]] = None,
        is_scope_data: bool = False,
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Gaussian distributions, which are:

        .. math::

            \text{supp}(\text{Gaussian})=(-\infty,+\infty)

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

        # NaN values do not throw an error but are simply flagged as False
        valid = self.dist(node_ids).support.check(scope_data)  # type: ignore

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # set nan_entries back to True
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf()

        return valid


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: GaussianLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[GaussianLayer, Gaussian, None]:
    """Structural marginalization for ``GaussianlLayer`` objects in the ``torch`` backend.

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
        return Gaussian(
            scope=marginalized_scopes[0],
            mean=layer.mean[node_id].item(),
            std=layer.std[node_id].item(),
        )
    else:
        return GaussianLayer(
            scope=marginalized_scopes,
            mean=layer.mean[marginalized_node_ids].detach(),
            std=layer.std[marginalized_node_ids].detach(),
        )


@dispatch(memoize=True)  # type: ignore
def toTorch(
    layer: BaseGaussianLayer, dispatch_ctx: Optional[DispatchContext] = None
) -> GaussianLayer:
    """Conversion for ``GaussianLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return GaussianLayer(scope=layer.scopes_out, mean=layer.mean, std=layer.std)


@dispatch(memoize=True)  # type: ignore
def toBase(
    layer: GaussianLayer, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseGaussianLayer:
    """Conversion for ``GaussianLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseGaussianLayer(
        scope=layer.scopes_out,
        mean=layer.mean.detach().numpy(),
        std=layer.std.detach().numpy(),
    )
