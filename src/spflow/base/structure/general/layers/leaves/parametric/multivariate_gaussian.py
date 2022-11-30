"""Contains Multivariate Gaussian leaf layer for SPFlow in the ``base`` backend.
"""
from typing import Iterable, List, Optional, Tuple, Type, Union

import numpy as np
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.base.structure.general.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.general.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian,
)
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


class MultivariateGaussianLayer(Module):
    r"""Layer of multiple multivariate Gaussian distribution leaf node in the ``base`` backend.

    Represents multiple multivariate Gaussian distributions with independent scopes, each with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{(2\pi)^d\det\Sigma}}\exp\left(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)\right)

    where
        - :math:`d` is the dimension of the distribution
        - :math:`x` is the :math:`d`-dim. vector of observations
        - :math:`\mu` is the :math:`d`-dim. mean vector
        - :math:`\Sigma` is the :math:`d\times d` covariance matrix

    Attributes:
        mean:
            List of one-dimensional NumPy array representing the means (:math:`\mu`) of each of the one-dimensional Gaussian distributions.
            Each row corresponds to a distribution.
        cov:
            List of two-dimensional NumPy array (representing :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
            of the scope of the respective distribution) describing the covariances of the distributions. The diagonals hold the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
            Each element of the first dimension corresponds to a distribution.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``MultivariateGaussian`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        mean: Optional[Union[List[float], List[List[float]], List[np.ndarray]]] = None,
        cov: Optional[Union[List[List[float]], List[List[List[float]]], List[np.ndarray]]] = None,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``MultivariateGaussianLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            mean:
                A list of floats, a list of lists of floats, a one-dimensional NumPy array or a list of one-dimensional NumPy array representing the means (:math:`\mu`) of each of the one-dimensional Gaussian distributions.
                Each row corresponds to a distribution. If a list of floats or one-dimensional NumPy array is given, it is broadcast to all nodes.
                Defaults to None, in which case all distributions are initialized with all zero means.
            cov:
                A list of lists of floats, a list of lists of lists of floats, a two-dimensional NumPy array or a list of two-dimensional NumPy arrays (representing :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
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
                raise ValueError("List of scopes for 'MultivariateGaussianLayer' was empty.")

            self._n_out = len(scope)

        super().__init__(children=[], **kwargs)

        if mean is None:
            mean = [np.zeros(len(s.query)) for s in scope]
        if cov is None:
            cov = [np.eye(len(s.query)) for s in scope]

        # create leaf nodes
        self.nodes = [MultivariateGaussian(s) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(mean, cov)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def mean(self) -> List[np.ndarray]:
        """Returns the means of the represented distributions."""
        return [node.mean for node in self.nodes]

    @property
    def cov(self) -> List[np.ndarray]:
        """Returns the covariance matrices of the represented distributions."""
        return [node.cov for node in self.nodes]

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``MultivariateGaussianLayer`` can represent one or more multivariate nodes with ``MetaType.Continuous`` or ``GaussianType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not MultivariateGaussian.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "MultivariateGaussianLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``MultivariateGaussianLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'MultivariateGaussianLayer' cannot be instantiated from the following signatures: {signatures}."
            )

        mean_list = []
        cov_list = []
        scopes = []

        for feature_ctx in signatures:

            mean, cov = np.zeros(len(feature_ctx.scope.query)), np.eye(len(feature_ctx.scope.query))

            for i, domain in enumerate(feature_ctx.get_domains()):
                # read or initialize parameters
                if domain == MetaType.Continuous:
                    pass
                elif domain == FeatureTypes.Gaussian:
                    # instantiate object
                    domain = domain()
                    mean[i], cov[i][i] = domain.mean, domain.std
                elif isinstance(domain, FeatureTypes.Gaussian):
                    mean[i], cov[i][i] = domain.mean, domain.std
                else:
                    raise ValueError(
                        f"Unknown signature type {domain} for 'MultivariateGaussianLayer' that was not caught during acception checking."
                    )

            mean_list.append(mean)
            cov_list.append(cov)
            scopes.append(feature_ctx.scope)

        return MultivariateGaussianLayer(scopes, mean=mean_list, cov=cov_list)

    def set_params(
        self,
        mean: Optional[Union[List[float], np.ndarray, List[List[float]], List[np.ndarray]]] = None,
        cov: Optional[
            Union[
                List[List[float]],
                np.ndarray,
                List[List[List[float]]],
                List[np.ndarray],
            ]
        ] = None,
    ) -> None:
        r"""Sets the parameters for the represented distributions.

        Args:
            mean:
                A list of floats, a list of lists of floats, a one-dimensional NumPy array or a list of one-dimensional NumPy array representing the means (:math:`\mu`) of each of the one-dimensional Gaussian distributions.
                Each row corresponds to a distribution. If a list of floats or one-dimensional NumPy array is given, it is broadcast to all nodes.
                Defaults to None, in which case all distributions are initialized with all zero means.
            cov:
                A list of lists of floats, a list of lists of lists of floats, a two-dimensional NumPy array or a list of two-dimensional NumPy arrays (representing :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
                of the scope of the respective distribution) describing the covariances of the distributions. The diagonals hold the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
                Each element of the first dimension corresponds to a distribution. If a list of lists of floats or two-dimensional NumPy array is given, it is broadcast to all nodes.
                Defaults to None, in which case all distributions are initialized with identity matrices.
        """
        if isinstance(mean, list):
            # can be a list of values specifying a single mean (broadcast to all nodes)
            if all([isinstance(m, float) or isinstance(m, int) for m in mean]):
                mean = [np.array(mean) for _ in range(self.n_out)]
            # can also be a list of different means
            else:
                mean = [m if isinstance(m, np.ndarray) else np.array(m) for m in mean]
        elif isinstance(mean, np.ndarray):
            # can be a one-dimensional numpy array specifying single mean (broadcast to all nodes)
            if mean.ndim == 1:
                mean = [mean for _ in range(self.n_out)]
            # can also be an array of different means
            else:
                mean = [m for m in mean]
        else:
            raise ValueError(f"Specified 'mean' for 'MultivariateGaussianLayer' is of unknown type {type(mean)}.")

        if isinstance(cov, list):
            # can be a list of lists of values specifying a single cov (broadcast to all nodes)
            if all([all([isinstance(c, float) or isinstance(c, int) for c in l]) for l in cov]):
                cov = [np.array(cov) for _ in range(self.n_out)]
            # can also be a list of different covs
            else:
                cov = [c if isinstance(c, np.ndarray) else np.array(c) for c in cov]
        elif isinstance(cov, np.ndarray):
            # can be a two-dimensional numpy array specifying single cov (broadcast to all nodes)
            if cov.ndim == 2:
                cov = [cov for _ in range(self.n_out)]
            # can also be an array of different covs
            else:
                cov = [c for c in cov]
        else:
            raise ValueError(f"Specified 'cov' for 'MultivariateGaussianLayer' is of unknown type {type(cov)}.")

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
                    f"All numpy arrays of 'mean' values for 'MultivariateGaussianLayer' are expected to be one-dimensional, but at least one is {m.ndim}-dimensional."
                )
            if m.shape[0] != len(s.query):
                raise ValueError(
                    f"Dimensions of a mean vector for 'MultivariateGaussianLayer' do not match corresponding scope size."
                )

            if c.ndim != 2:
                raise ValueError(
                    f"All numpy arrays of 'cov' values for 'MultivariateGaussianLayer' are expected to be two-dimensional, but at least one is {c.ndim}-dimensional."
                )
            if c.shape[0] != len(s.query) or c.shape[1] != len(s.query):
                raise ValueError(
                    f"Dimensions of a covariance matrix for 'MultivariateGaussianLayer' do not match corresponding scope size."
                )

        for node_mean, node_cov, node in zip(mean, cov, self.nodes):
            node.set_params(node_mean, node_cov)

    def get_params(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of a list of one-dimensional NumPy array and a list of a two-dimensional NumPy array representing the means and covariances, respectively.
        """
        return self.mean, self.cov

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

    def check_support(self, data: np.ndarray, node_ids: Optional[List[int]] = None) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Multivariate Gaussian distributions, which are:

        .. math::

            \text{supp}(\text{MultivariateGaussian})=(-\infty,+\infty)^k

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

        return np.concatenate([self.nodes[i].check_support(data) for i in node_ids], axis=1)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: MultivariateGaussianLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[MultivariateGaussianLayer, MultivariateGaussian, Gaussian, None]:
    r"""Structural marginalization for ``MultivariateGaussianLayer`` objects in the ``base`` backend.

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
        new_layer = MultivariateGaussianLayer(marg_scopes, *[np.array(p) for p in zip(*marg_params)])
        return new_layer
