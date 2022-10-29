# -*- coding: utf-8 -*-
"""Contains Multivariate Gaussian leaf layer for SPFlow in the ``torch`` backend.
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
from spflow.torch.structure.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian,
)
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.layers.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussianLayer as BaseMultivariateGaussianLayer,
)


class MultivariateGaussianLayer(Module):
    r"""Layer of multiple multivariate Gaussian distribution leaf node in the ``torch`` backend.

    Represents multiple multivariate Gaussian distributions with independent scopes, each with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{(2\pi)^d\det\Sigma}}\exp\left(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)\right)

    where
        - :math:`d` is the dimension of the distribution
        - :math:`x` is the :math:`d`-dim. vector of observations
        - :math:`\mu` is the :math:`d`-dim. mean vector
        - :math:`\Sigma` is the :math:`d\times d` covariance matrix

    In contrast to other layers is ``torch`` backend, the layer is composed of ``MultivariateGaussian`` nodes and is not further optimized.
    Note, that different to ``MultivariateGaussianLayer`` in the ``base`` backend, the ``torch`` implementation only accepts positive definite (as opposed to positive semi-definite) covariance matrices.

    Attributes:
        mean:
            List of one-dimensional PyTorch tensors representing the means (:math:`\mu`) of each distribution.
            Each row corresponds to a distribution.
        cov:
            List of two-dimensional PyTorch tensors (representing :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
            of the scope of the respective distribution) describing the covariances of the distributions. The diagonals hold the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
        nodes:
            List of ``MultivariateGaussian`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        mean: Optional[
            Union[List[float], List[List[float]], np.ndarray, torch.Tensor]
        ] = None,
        cov: Optional[
            Union[
                List[List[float]],
                List[List[List[float]]],
                np.ndarray,
                torch.Tensor,
            ]
        ] = None,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``MultivariateGaussianLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            mean:
                A list of floats, a list of lists of floats, a one-dimensional NumPy array or PyTorch tensor or a list of one-dimensional NumPy arrays or PyTorch tensors representing the means (:math:`\mu`) of each of the one-dimensional Gaussian distributions.
                Each row corresponds to a distribution. If a list of floats or one-dimensional NumPy array is given, it is broadcast to all nodes.
                Defaults to None, in which case all distributions are initialized with all zero means.
            cov:
                A list of lists of floats, a list of lists of lists of floats, a two-dimensional NumPy array or PyTorch tensor or a list of two-dimensional NumPy arrays or PyTorch tensors (representing :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
                of the scope of the respective distribution) describing the covariances of the distributions. The diagonals hold the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
                Each element of the first dimension corresponds to a distribution. If a list of lists of floats or two-dimensional NumPy array is given, it is broadcast to all nodes.
                Defaults to None, in which case all distributions are initialized with identity matrices.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'MultivariateGaussianLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError(
                    "List of scopes for 'MultivariateGaussianLayer' was empty."
                )

            self._n_out = len(scope)

        super(MultivariateGaussianLayer, self).__init__(children=[], **kwargs)

        if mean is None:
            mean = [torch.zeros(len(s.query)) for s in scope]
        if cov is None:
            cov = [torch.eye(len(s.query)) for s in scope]

        # create leaf nodes
        self.nodes = torch.nn.ModuleList(
            [MultivariateGaussian(s) for s in scope]
        )

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(
            lambda s1, s2: s1.union(s2), self.scopes_out
        )

        # parse weights
        self.set_params(mean, cov)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def mean(self) -> List[np.ndarray]:
        """TODO"""
        return [node.mean for node in self.nodes]

    @property
    def cov(self) -> List[np.ndarray]:
        """TODO"""
        return [node.cov for node in self.nodes]

    def dist(
        self, node_ids: Optional[List[int]] = None
    ) -> List[D.Distribution]:
        r"""Returns the PyTorch distributions represented by the leaf layer.

        Args:
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            List of ``torch.distributions.MultivariateNormal`` instances.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return [self.nodes[i].dist for i in node_ids]

    def set_params(
        self,
        mean: Optional[
            Union[
                List[float],
                np.ndarray,
                torch.Tensor,
                List[List[float]],
                List[np.ndarray],
                List[torch.Tensor],
            ]
        ] = None,
        cov: Optional[
            Union[
                List[List[float]],
                np.ndarray,
                torch.Tensor,
                List[List[List[float]]],
                List[np.ndarray],
                List[torch.Tensor],
            ]
        ] = None,
    ) -> None:
        r"""Sets the parameters for the represented distributions.

        Args:
            mean:
                A list of floats, a list of lists of floats, a one-dimensional NumPy array or PyTorch tensor or a list of one-dimensional NumPy arrays or PyTorch tensors representing the means (:math:`\mu`) of each of the one-dimensional Gaussian distributions.
                Each row corresponds to a distribution. If a list of floats or one-dimensional NumPy array is given, it is broadcast to all nodes.
            cov:
                A list of lists of floats, a list of lists of lists of floats, a two-dimensional NumPy array or PyTorch tensor or a list of two-dimensional NumPy arrays or PyTorch tensors (representing :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
                of the scope of the respective distribution) describing the covariances of the distributions. The diagonals hold the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
                Each element of the first dimension corresponds to a distribution. If a list of lists of floats or two-dimensional NumPy array is given, it is broadcast to all nodes.
        """
        if isinstance(mean, list):
            # can be a list of values specifying a single mean (broadcast to all nodes)
            if all([isinstance(m, float) or isinstance(m, int) for m in mean]):
                mean = [np.array(mean) for _ in range(self.n_out)]
            # can also be a list of different means
            else:
                mean = [
                    m if isinstance(m, np.ndarray) else np.array(m)
                    for m in mean
                ]
        elif isinstance(mean, np.ndarray) or isinstance(mean, torch.Tensor):
            # can be a one-dimensional numpy array/torch tensor specifying single mean (broadcast to all nodes)
            if mean.ndim == 1:
                mean = [mean for _ in range(self.n_out)]
            # can also be an array of different means
            else:
                mean = [m for m in mean]
        else:
            raise ValueError(
                f"Specified 'mean' for 'MultivariateGaussianLayer' is of unknown type {type(mean)}."
            )

        if isinstance(cov, list):
            # can be a list of lists of values specifying a single cov (broadcast to all nodes)
            if all(
                [
                    all([isinstance(c, float) or isinstance(c, int) for c in l])
                    for l in cov
                ]
            ):
                cov = [np.array(cov) for _ in range(self.n_out)]
            # can also be a list of different covs
            else:
                cov = [
                    c if isinstance(c, np.ndarray) else np.array(c) for c in cov
                ]
        elif isinstance(cov, np.ndarray) or isinstance(cov, torch.Tensor):
            # can be a two-dimensional numpy array/torch tensor specifying single cov (broadcast to all nodes)
            if cov.ndim == 2:
                cov = [cov for _ in range(self.n_out)]
            # can also be an array of different covs
            else:
                cov = [c for c in cov]
        else:
            raise ValueError(
                f"Specified 'cov' for 'MultivariateGaussianLayer' is of unknown type {type(cov)}."
            )

        if len(mean) != self.n_out:
            raise ValueError(
                f"Length of list of 'mean' values for 'MultivariateGaussianLayer' must match number of output nodes {self.n_out}, but is {len(mean)}"
            )
        if len(cov) != self.n_out:
            raise ValueError(
                f"Length of list of 'cov' values for 'MultivariateGaussianLayer' must match number of output nodes {self.n_out}, but is {len(cov)}"
            )

        for m, c, s in zip(mean, cov, self.scopes_out):
            if m.ndim != 1:
                raise ValueError(
                    f"All tensors of 'mean' values for 'MultivariateGaussianLayer' are expected to be one-dimensional, but at least one is {m.ndim}-dimensional."
                )
            if m.shape[0] != len(s.query):
                raise ValueError(
                    f"Dimensions of a mean vector for 'MultivariateGaussianLayer' do not match corresponding scope size."
                )

            if c.ndim != 2:
                raise ValueError(
                    f"All tensors of 'cov' values for 'MultivariateGaussianLayer' are expected to be two-dimensional, but at least one is {c.ndim}-dimensional."
                )
            if c.shape[0] != len(s.query) or c.shape[1] != len(s.query):
                raise ValueError(
                    f"Dimensions of a covariance matrix for 'MultivariateGaussianLayer' do not match corresponding scope size."
                )

        for node_mean, node_cov, node in zip(mean, cov, self.nodes):
            node.set_params(node_mean, node_cov)

    def get_params(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of a list of one-dimensional PyTorch tensor and a list of a two-dimensional PyTorch tensor representing the means and covariances, respectively.
        """
        return (self.mean, self.cov)

    def check_support(
        self, data: torch.Tensor, node_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Multivariate Gaussian distributions, which are:

        .. math::

            \text{supp}(\text{MultivariateGaussian})=(-\infty,+\infty)^k

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            TODO
            scope_data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.

        Returns:
            Two dimensional PyTorch tensor indicating for each instance and node, whether they are part of the support (True) or not (False).
            Each row corresponds to an input sample.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return torch.concat(
            [self.nodes[i].check_support(data) for i in node_ids], dim=1
        )


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: MultivariateGaussianLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[MultivariateGaussianLayer, MultivariateGaussian, Gaussian, None]:
    """Structural marginalization for ``MultivariateGaussianLayer`` objects in the ``torch`` backend.

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

    # marginalize nodes
    marg_nodes = []
    marg_scopes = []
    marg_params = []

    for node in layer.nodes:
        marg_node = marginalize(node, marg_rvs, prune=prune)

        if marg_node is not None:
            marg_scopes.append(marg_node.scope)
            marg_params.append(marg_node.get_params())
            marg_nodes.append(marg_node)

    if len(marg_scopes) == 0:
        return None
    elif len(marg_scopes) == 1 and prune:
        return marg_nodes.pop()
    else:
        new_layer = MultivariateGaussianLayer(
            marg_scopes, *[list(p) for p in zip(*marg_params)]
        )
        return new_layer


@dispatch(memoize=True)  # type: ignore
def toTorch(
    layer: BaseMultivariateGaussianLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> MultivariateGaussianLayer:
    """Conversion for ``MultivariateGaussianLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return MultivariateGaussianLayer(
        scope=layer.scopes_out, mean=layer.mean, cov=layer.cov
    )


@dispatch(memoize=True)  # type: ignore
def toBase(
    layer: MultivariateGaussianLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> BaseMultivariateGaussianLayer:
    """Conversion for ``MultivariateGaussianLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseMultivariateGaussianLayer(
        scope=layer.scopes_out,
        mean=[m.detach().numpy() for m in layer.mean],
        cov=[c.detach().numpy() for c in layer.cov],
    )
