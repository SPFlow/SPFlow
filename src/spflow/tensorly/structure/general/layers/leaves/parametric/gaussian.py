"""Contains Gaussian leaf layer for SPFlow in the ``base`` backend.
"""
from typing import Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import tensorly as tl
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.base.structure.general.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.module import Module
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureType, FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class GaussianLayer(Module):
    r"""Layer of multiple (univariate) Gaussian distribution leaf nodes in the ``base`` backend.

    Represents multiple univariate Gaussian distributions with independent scopes, each with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})

    where
        - :math:`x` the observation
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
            List of ``Gaussian`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        mean: Union[float, List[float], tl.tensor] = 0.0,
        std: Union[float, List[float], tl.tensor] = 1.0,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``GaussianLayer`` object.

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
                    f"Number of nodes for 'GaussianLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'GaussianLayer' was empty.")

            self._n_out = len(scope)

        super().__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [Gaussian(s, 0.0, 1.0) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(mean, std)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def mean(self) -> tl.tensor:
        """Returns the means of the represented distributions."""
        return tl.tensor([node.mean for node in self.nodes])

    @property
    def std(self) -> tl.tensor:
        """Returns the standard deviations of the represented distributions."""
        return tl.tensor([node.std for node in self.nodes])

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``GaussianLayer`` can represent one or more univariate nodes with ``MetaType.Continuous`` or ``GaussianType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not Gaussian.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "GaussianLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``GaussianLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'GaussianLayer' cannot be instantiated from the following signatures: {signatures}.")

        mean = []
        std = []
        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if domain == MetaType.Continuous:
                mean.append(0.0)
                std.append(1.0)
            elif domain == FeatureTypes.Gaussian:
                # instantiate object
                domain = domain()
                mean.append(domain.mean)
                std.append(domain.std)
            elif isinstance(domain, FeatureTypes.Gaussian):
                mean.append(domain.mean)
                std.append(domain.std)
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'GaussianLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return GaussianLayer(scopes, mean=mean, std=std)

    def set_params(
        self,
        mean: Union[int, float, List[float], tl.tensor] = 0.0,
        std: Union[int, float, List[float], tl.tensor] = 1.0,
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
            mean = tl.tensor([mean for _ in range(self.n_out)])
        if isinstance(mean, list):
            mean = tl.tensor(mean)
        if tl.ndim(mean) != 1:
            raise ValueError(
                f"Numpy array of 'mean' values for 'GaussianLayer' is expected to be one-dimensional, but is {tl.ndim(mean)}-dimensional."
            )
        if mean.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'mean' values for 'GaussianLayer' must match number of output nodes {self.n_out}, but is {mean.shape[0]}"
            )

        if isinstance(std, int) or isinstance(std, float):
            std = tl.tensor([float(std) for _ in range(self.n_out)])
        if isinstance(std, list):
            std = tl.tensor(std)
        if tl.ndim(std) != 1:
            raise ValueError(
                f"Numpy array of 'std' values for 'GaussianLayer' is expected to be one-dimensional, but is {tl.ndim(std)}-dimensional."
            )
        if std.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'std' values for 'GaussianLayer' must match number of output nodes {self.n_out}, but is {std.shape[0]}"
            )

        for node_mean, node_std, node in zip(mean, std, self.nodes):
            node.set_params(node_mean, node_std)

    def get_params(self) -> Tuple[tl.tensor, tl.tensor]:
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

    def check_support(self, data: tl.tensor, node_ids: Optional[List[int]] = None) -> tl.tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Gaussian distributions, which are:

        .. math::

            \text{supp}(\text{Gaussian})=(-\infty,+\infty)

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

        return tl.concatenate([self.nodes[i].check_support(data) for i in node_ids], axis=1)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: GaussianLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[GaussianLayer, Gaussian, None]:
    r"""Structural marginalization for ``GaussianLayer`` objects in the ``base`` backend.

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
        new_node = Gaussian(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = GaussianLayer(marg_scopes, *[tl.tensor(p) for p in zip(*marg_params)])
        return new_layer
