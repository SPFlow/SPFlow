# -*- coding: utf-8 -*-
"""Contains conditional Multivariate Normal leaf node for SPFlow in the ``base`` backend.
"""
from typing import Tuple, List, Union, Optional, Iterable, Union, Callable, Type
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import MetaType, FeatureType, FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.cond_gaussian import (
    CondGaussian,
)

from scipy.stats import multivariate_normal  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class CondMultivariateGaussian(LeafNode):
    r"""Conditional Multivariate Gaussian distribution leaf node in the ``base`` backend.

    Represents a conditional multivariate Gaussian distribution, with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{(2\pi)^d\det\Sigma}}\exp\left(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)\right)

    where
        - :math:`d` is the dimension of the distribution
        - :math:`x` is the :math:`d`-dim. vector of observations
        - :math:`\mu` is the :math:`d`-dim. mean vector
        - :math:`\Sigma` is the :math:`d\times d` covariance matrix

    Attributes
        cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``mean``,``cov`` as keys.
            The value for ``mean`` should be a list of floating point values or one-dimensional NumPy array
            containing the means.
            The value for ``cov`` should be a list of lists of floating points or two-dimensional NumPy array
            containing a symmetric positive semi-definite matrix.
    """

    def __init__(
        self,
        scope: Scope,
        cond_f: Optional[Callable] = None,
    ) -> None:
        r"""Initializes ``CondMultivariateGaussian`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``mean``,``cov`` as keys.
                The value for ``mean`` should be a list of floating point values or one-dimensional NumPy array
                containing the means.
                The value for ``cov`` should be a list of lists of floating points or two-dimensional NumPy array
                containing a symmetric positive semi-definite matrix.
        """
        # check if scope contains duplicates
        if len(set(scope.query)) != len(scope.query):
            raise ValueError(
                "Query scope for 'CondMultivariateGaussian' contains duplicate variables."
            )
        if len(scope.evidence) == 0:
            raise ValueError(
                f"Evidence scope for 'CondMultivariateGaussian' should not be empty."
            )
        if len(scope.query) < 1:
            raise ValueError(
                "Size of query scope for 'CondMultivariateGaussian' must be at least 1."
            )

        super(CondMultivariateGaussian, self).__init__(scope=scope)

        # set optional conditional function
        self.set_cond_f(cond_f)

    @classmethod
    def accepts(self, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CondMultivariateGaussian`` can represent a single multivariate node with ``MetaType.Continuous`` or ``GaussianType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        feature_ctx = signatures[0]
        domains = feature_ctx.get_domains()

        # leaf is a single non-conditional (possibly multivariate) node
        if (
            len(domains) < 1
            or len(feature_ctx.scope.query) != len(domains)
            or len(feature_ctx.scope.evidence) == 0
        ):
            return False

        # leaf is a continuous (multivariate) Gaussian distribution
        if not all(
            [
                domain == FeatureTypes.Continuous
                or domain == FeatureTypes.Gaussian
                or isinstance(domain, FeatureTypes.Gaussian)
                for domain in domains
            ]
        ):
            return False

        return True

    @classmethod
    def from_signatures(
        self, signatures: List[FeatureContext]
    ) -> "CondMultivariateGaussian":
        """Creates an instance from a specified signature.

        Returns:
            ``CondMultivariateGaussian`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not self.accepts(signatures):
            raise ValueError(
                f"'CondMultivariateGaussian' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]

        for domain in feature_ctx.get_domains():
            # read or initialize parameters
            if (
                domain == MetaType.Continuous
                or domain == FeatureTypes.Gaussian
                or isinstance(domain, FeatureTypes.Gaussian)
            ):
                pass
            else:
                raise ValueError(
                    f"Unknown signature type {type} for 'CondMultivariateGaussian' that was not caught during acception checking."
                )

        return CondMultivariateGaussian(feature_ctx.scope)

    def set_cond_f(self, cond_f: Optional[Callable] = None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``mean``,``cov`` as keys.
                The value for ``mean`` should be a list of floating point values or one-dimensional NumPy array
                containing the means.
                The value for ``cov`` should be a list of lists of floating points or two-dimensional NumPy array
                containing a symmetric positive semi-definite matrix.
        """
        self.cond_f = cond_f

    def retrieve_params(
        self, data: np.ndarray, dispatch_ctx: DispatchContext
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Retrieves the conditional parameter of the leaf node.

        First, checks if conditional parameters (``mean``,``cov``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameters.

        Args:
            data:
                Two-dimensional NumPy array containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            Tuple of a one- and a two-dimensional NumPy array representing the mean and covariance matrix.

        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
        mean, cov, cond_f = None, None, None

        # check dispatch cache for required conditional parameters 'mean', 'cov'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if values for 'mean', 'cov' are specified (highest priority)
            if "mean" in args:
                mean = args["mean"]
            if "cov" in args:
                cov = args["cov"]
            # check if alternative function to provide 'mean', 'cov' is specified (second to highest priority)
            if "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'mean','cov' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'mean' or 'cov' nor 'cond_f' is specified (via node or arguments)
        if (mean is None or cov is None) and cond_f is None:
            raise ValueError(
                "'CondMultivariateGaussian' requires either 'mean' and 'cov' or 'cond_f' to retrieve 'mean', 'cov' to be specified."
            )

        # if 'mean' or 'cov' not already specified, retrieve them
        if mean is None or cov is None:
            params = cond_f(data)
            mean = params["mean"]
            cov = params["cov"]

        # cast lists to numpy arrays
        if isinstance(mean, List):
            mean = np.array(mean)
        if isinstance(cov, List):
            cov = np.array(cov)

        # check mean vector dimensions
        if (
            (mean.ndim == 1 and mean.shape[0] != len(self.scope.query))
            or (mean.ndim == 2 and mean.shape[1] != len(self.scope.query))
            or mean.ndim > 2
        ):
            raise ValueError(
                f"Dimensions of 'mean' for 'CondMultivariateGaussian' should match scope size {len(self.scope.query)}, but was: {mean.shape}."
            )

        if mean.ndim == 2:
            mean = mean.squeeze(0)

        # check mean vector for nan or inf values
        if np.any(np.isinf(mean)):
            raise ValueError(
                "Value of 'mean' for 'CondMultivariateGaussian' may not contain infinite values."
            )
        if np.any(np.isnan(mean)):
            raise ValueError(
                "Value of 'mean' for 'CondMultivariateGaussian' may not contain NaN values."
            )

        # test whether or not matrix has correct shape
        if cov.ndim != 2 or (
            cov.ndim == 2
            and (
                cov.shape[0] != len(self.scope.query)
                or cov.shape[1] != len(self.scope.query)
            )
        ):
            raise ValueError(
                f"Value of 'cov' for 'CondMultivariateGaussian' expected to be of shape ({len(self.scope.query), len(self.scope.query)}), but was: {cov.shape}."
            )

        # check covariance matrix for nan or inf values
        if np.any(np.isinf(cov)):
            raise ValueError(
                "Value of 'cov' for 'CondMultivariateGaussian' may not contain infinite values."
            )
        if np.any(np.isnan(cov)):
            raise ValueError(
                "Value of 'cov' for 'CondMultivariateGaussian' may not contain NaN values."
            )

        # test covariance matrix for symmetry
        if not np.all(cov == cov.T):
            raise ValueError(
                "Value of 'cov' for 'CondMultivariateGaussian' must be symmetric."
            )

        # test covariance matrix for positive semi-definiteness
        # NOTE: since we established in the test right before that matrix is symmetric we can use numpy's eigvalsh instead of eigvals
        if np.any(np.linalg.eigvalsh(cov) < 0):
            raise ValueError(
                "Value of 'cov' for 'CondMultivariateGaussian' must be positive semi-definite."
            )

        return mean, cov

    def dist(self, mean: np.ndarray, cov: np.ndarray) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Args:
            mean:
                A list of floating points or one-dimensional NumPy array containing the means (:math:`\mu`) of each of the one-dimensional Normal distributions.
                Must have exactly as many elements as the scope of this leaf.
                Defaults to all zeros.
            cov:
                A list of lists of floating points or a two-dimensional NumPy array (representing a :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
                of the scope) describing the covariances of the distribution. The diagonal holds the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
                Defaults to the identity matrix.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return multivariate_normal(mean=mean, cov=cov)

    def check_support(
        self, data: np.ndarray, is_scope_data: bool = False
    ) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Multivariate Gaussian distribution, which is:

        .. math::

            \text{supp}(\text{MultivariateGaussian})=(-\infty,+\infty)^k

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional NumPy array containing sample instances.
                Each row is regarded as a sample.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leaf's scope in the correct order (True) or if it needs to be extracted from the full data set.
                Defaults to False.

        Returns:
            Two-dimensional NumPy array indicating for each instance, whether they are part of the support (True) or not (False).
        """
        if is_scope_data:
            scope_data = data
        else:
            # select relevant data for scope
            scope_data = data[:, self.scope.query]

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope.query):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scope.query)}), but was: {scope_data.shape}"
            )

        valid = np.ones(scope_data.shape, dtype=bool)

        # nan entries (regarded as valid)
        nan_mask = np.isnan(scope_data)

        # check for infinite values
        # additionally check for infinite values (may return NaNs despite support)
        valid[~nan_mask] &= ~np.isinf(scope_data[~nan_mask])

        return valid


@dispatch(memoize=True)  # type: ignore
def marginalize(
    node: CondMultivariateGaussian,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[CondMultivariateGaussian, CondGaussian, None]:
    """Structural marginalization for ``CondMultivariateGaussian`` nodes in the ``base`` backend.

    Structurally marginalizes the leaf node.
    If the node's scope contains non of the random variables to marginalize, then the node is returned unaltered.
    If the node's scope is fully marginalized over, then None is returned.
    If the node's scope is partially marginalized over, a marginal uni- or multivariate Gaussian is returned instead.

    Args:
        node:
            Node module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect here. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
            Unaltered node if module is not marginalized, marginalized uni- or multivariate Gaussian leaf node, or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    scope = node.scope

    # scope after marginalization (important: must remain order of scope indices since they map to the indices of the mean vector and covariance matrix!)
    marg_scope = []
    marg_scope_ids = []

    for rv in scope.query:
        if rv not in marg_rvs:
            marg_scope.append(rv)
            marg_scope_ids.append(scope.query.index(rv))

    # return univariate Gaussian if one-dimensional
    if len(marg_scope) == 1:
        # note: Gaussian requires standard deviations instead of variance (take square root)
        return CondGaussian(Scope(marg_scope, scope.evidence))
    # entire node is marginalized over
    elif len(marg_scope) == 0:
        return None
    # node is partially marginalized over
    else:
        # compute marginalized mean vector and covariance matrix
        marg_scope_ids = [scope.query.index(rv) for rv in marg_scope]

        return CondMultivariateGaussian(Scope(marg_scope, scope.evidence))
