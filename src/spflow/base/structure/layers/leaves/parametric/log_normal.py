# -*- coding: utf-8 -*-
"""Contains Log-Normal leaf layer for SPFlow in the ``base`` backend.
"""
from typing import List, Union, Optional, Iterable, Tuple
import numpy as np
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.scope.scope import Scope
from spflow.base.structure.module import Module
from spflow.base.structure.nodes.leaves.parametric.log_normal import LogNormal


class LogNormalLayer(Module):
    r"""Layer of multiple (univariate) Log-Normal distribution leaf nodes in the ``base`` backend.

    Represents multiple univariate Log-Normal distributions with independent scopes, each with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln(x)-\mu)^2}{2\sigma^2}\right)

    where
        - :math:`x` is an observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Attributes:
        mean:
            One-dimensional NumPy array representing the means (:math:`\mu`) of the distributions.
        std:
            One-dimensional NumPy array representing the standard deviations (:math:`\sigma`) of the distributions (must be greater than 0).
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``LogNormal`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        mean: Union[float, List[float], np.ndarray] = 0.0,
        std: Union[float, List[float], np.ndarray] = 1.0,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``LogNormalLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            mean:
                Floating point, list of floats or one-dimensional NumPy array representing the means (:math:`\mu`).
                If a single floating point value is given it is broadcast to all nodes.
                Defaults to 0.0.
            std:
                Floating point, list of floats or one-dimensional NumPy array representing the standard deviations (:math:`\sigma`), greater than 0.
                If a single floating point value is given it is broadcast to all nodes.
                Defaults to 1.0.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'LogNormalLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError(
                    "List of scopes for 'LogNormalLayer' was empty."
                )

            self._n_out = len(scope)

        super(LogNormalLayer, self).__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [LogNormal(s, 0.0, 1.0) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(mean, std)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def mean(self) -> np.ndarray:
        """Returns the means of the represented distributions."""
        return np.array([node.mean for node in self.nodes])

    @property
    def std(self) -> np.ndarray:
        """Returns the standard deviations of the represented distributions."""
        return np.array([node.std for node in self.nodes])

    def set_params(
        self,
        mean: Union[int, float, List[float], np.ndarray] = 0.0,
        std: Union[int, float, List[float], np.ndarray] = 1.0,
    ) -> None:
        r"""Sets the parameters for the represented distributions.

        Args:
            mean:
                Floating point, list of floats or one-dimensional NumPy array representing the means (:math:`\mu`).
                If a single floating point value is given it is broadcast to all nodes.
                Defaults to 0.0.
            std:
                Floating point, list of floats or one-dimensional NumPy array representing the standard deviations (:math:`\sigma`), greater than 0.
                If a single floating point value is given it is broadcast to all nodes.
                Defaults to 1.0.
        """
        if isinstance(mean, int) or isinstance(mean, float):
            mean = np.array([mean for _ in range(self.n_out)])
        if isinstance(mean, list):
            mean = np.array(mean)
        if mean.ndim != 1:
            raise ValueError(
                f"Numpy array of 'mean' values for 'LogNormalLayer' is expected to be one-dimensional, but is {mean.ndim}-dimensional."
            )
        if mean.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'mean' values for 'LogNormalLayer' must match number of output nodes {self.n_out}, but is {mean.shape[0]}"
            )

        if isinstance(std, int) or isinstance(std, float):
            std = np.array([float(std) for _ in range(self.n_out)])
        if isinstance(std, list):
            std = np.array(std)
        if std.ndim != 1:
            raise ValueError(
                f"Numpy array of 'std' values for 'LogNormalLayer' is expected to be one-dimensional, but is {std.ndim}-dimensional."
            )
        if std.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'std' values for 'LogNormalLayer' must match number of output nodes {self.n_out}, but is {std.shape[0]}"
            )

        for node_mean, node_std, node in zip(mean, std, self.nodes):
            node.set_params(node_mean, node_std)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of one-dimensional NumPy arrays representing the means and standard deviations.
        """
        return self.mean, self.std

    def dist(self, node_ids: Optional[List[int]] = None) -> List[rv_frozen]:
        r"""Returns the SciPy distributions represented by the leaf layer.

        Args:
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            List of ``scipy.stats.distributions.rv_frozen`` distributions.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return [self.nodes[i].dist for i in node_ids]

    def check_support(
        self, data: np.ndarray, node_ids: Optional[List[int]] = None
    ) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Log-Normal distributions, which are:

        .. math::

            \text{supp}(\text{LogNormal})=(0,\infty)

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional NumPy array containing sample instances.
                Each row is regarded as a sample.
                Assumes that relevant data is located in the columns corresponding to the scope indices.
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            Two dimensional NumPy array indicating for each instance and node, whether they are part of the support (True) or not (False).
            Each row corresponds to an input sample.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return np.concatenate(
            [self.nodes[i].check_support(data) for i in node_ids], axis=1
        )


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: LogNormalLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[LogNormalLayer, LogNormal, None]:
    r"""Structural marginalization for ``LogNormalLayer`` objects in the ``base`` backend.

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
    marg_scopes = []
    marg_params = []

    for node in layer.nodes:
        marg_node = marginalize(node, marg_rvs, prune=prune)

        if marg_node is not None:
            marg_scopes.append(marg_node.scope)
            marg_params.append(marg_node.get_params())

    if len(marg_scopes) == 0:
        return None
    elif len(marg_scopes) == 1 and prune:
        new_node = LogNormal(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = LogNormalLayer(
            marg_scopes, *[np.array(p) for p in zip(*marg_params)]
        )
        return new_layer
