"""Contains Multivariate Normal leaf node for SPFlow in the ``base`` backend.
"""
from typing import Tuple, List, Union, Optional, Iterable
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import MetaType, FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.structure.general.nodes.leaf_node import LeafNode
from spflow.base.structure.general.nodes.leaves.parametric.gaussian import (
    Gaussian,
)

from scipy.stats import multivariate_normal  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class MultivariateGaussian(LeafNode):
    r"""Multivariate Gaussian distribution leaf node in the 'base' backend.

    Represents a multivariate Gaussian distribution, with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{(2\pi)^d\det\Sigma}}\exp\left(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)\right)

    where
        - :math:`d` is the dimension of the distribution
        - :math:`x` is the :math:`d`-dim. vector of observations
        - :math:`\mu` is the :math:`d`-dim. mean vector
        - :math:`\Sigma` is the :math:`d\times d` covariance matrix

    Attributes:
        mean:
            A list of floating points or one-dimensional NumPy array containing the means (:math:`\mu`) of each of the one-dimensional Gaussian distributions.
            Must have exactly as many elements as the scope of this leaf.
        cov:
            A list of lists of floating points or a two-dimensional NumPy array (representing a :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
            of the scope) describing the covariances of the distribution. The diagonal holds the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
    """

    def __init__(
        self,
        scope: Scope,
        mean: Optional[Union[List[float], np.ndarray]] = None,
        cov: Optional[Union[List[List[float]], np.ndarray]] = None,
    ) -> None:
        r"""Initializes ``MultivariateGaussian`` leaf node.

        Args:
            mean:
                A list of floating points or one-dimensional NumPy array containing the means (:math:`\mu`) of each of the one-dimensional Normal distributions.
                Must have exactly as many elements as the scope of this leaf.
                Defaults to all zeros.
            cov:
                A list of lists of floating points or a two-dimensional NumPy array (representing a :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
                of the scope) describing the covariances of the distribution. The diagonal holds the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
                Defaults to the identity matrix.
        """
        # check if scope contains duplicates
        if len(set(scope.query)) != len(scope.query):
            raise ValueError(
                "Query scope for 'MultivariateGaussian' contains duplicate variables."
            )
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'MultivariateGaussian' should be empty, but was {scope.evidence}."
            )
        if len(scope.query) < 1:
            raise ValueError(
                "Size of query scope for 'MultivariateGaussian' must be at least 1."
            )

        super().__init__(scope=scope)

        if mean is None:
            mean = np.zeros((1, len(scope.query)))
        if cov is None:
            cov = np.eye(len(scope.query))

        self.set_params(mean, cov)

    @classmethod
    def accepts(self, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``MultivariateGaussian`` can represent a single univariate node with ``MetaType.Continuous`` or ``GaussianType`` domains.

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
            or len(feature_ctx.scope.evidence) != 0
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
    ) -> "MultivariateGaussian":
        """Creates an instance from a specified signature.

        Returns:
            ``MultivariateGaussian`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not self.accepts(signatures):
            raise ValueError(
                f"'MultivariateGaussian' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]

        mean, cov = np.zeros(len(feature_ctx.scope.query)), np.eye(
            len(feature_ctx.scope.query)
        )

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
                    f"Unknown signature type {domain} for 'MultivariateGaussian' that was not caught during acception checking."
                )

        return MultivariateGaussian(feature_ctx.scope, mean=mean, cov=cov)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return multivariate_normal(mean=self.mean, cov=self.cov)

    def set_params(
        self,
        mean: Union[List[float], np.ndarray],
        cov: Union[List[List[float]], np.ndarray],
    ) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            mean:
                A list of floating points or one-dimensional NumPy array containing the means (:math:`\mu`) of each of the one-dimensional Normal distributions.
                Must have exactly as many elements as the scope of this leaf.
                Defaults to all zeros.
            cov:
                A list of lists of floating points or a two-dimensional NumPy array (representing a :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
                of the scope) describing the covariances of the distribution. The diagonal holds the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
                Defaults to the identity matrix.
        """
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
                f"Dimensions of 'mean' for 'MultivariateGaussian' should match scope size {len(self.scope.query)}, but was: {mean.shape}."
            )

        if mean.ndim == 2:
            mean = mean.squeeze(0)

        # check mean vector for nan or inf values
        if np.any(np.isinf(mean)):
            raise ValueError(
                "Value of 'mean' for 'MultivariateGaussian' may not contain infinite values."
            )
        if np.any(np.isnan(mean)):
            raise ValueError(
                "Value of 'mean' for 'MultivariateGaussian' may not contain NaN values."
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
                f"Value of 'cov' for 'MultivariateGaussian' expected to be of shape ({len(self.scope.query), len(self.scope.query)}), but was: {cov.shape}."
            )

        # check covariance matrix for nan or inf values
        if np.any(np.isinf(cov)):
            raise ValueError(
                "Value of 'cov' for 'MultivariateGaussian' may not contain infinite values."
            )
        if np.any(np.isnan(cov)):
            raise ValueError(
                "Value of 'cov' for 'MultivariateGaussian' may not contain NaN values."
            )

        # test covariance matrix for symmetry
        if not np.all(cov == cov.T):
            raise ValueError(
                "Value of 'cov' for 'MultivariateGaussian' must be symmetric."
            )

        # test covariance matrix for positive semi-definiteness
        # NOTE: since we established in the test right before that matrix is symmetric we can use numpy's eigvalsh instead of eigvals
        if np.any(np.linalg.eigvalsh(cov) < 0):
            raise ValueError(
                "Value of 'cov' for 'MultivariateGaussian' must be positive semi-definite."
            )

        self.mean = mean
        self.cov = cov

    def get_params(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of a one-dimensional and a two-dimensional NumPy array representing the mean and covariance matrix, respectively.
        """
        return self.mean, self.cov

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
    node: MultivariateGaussian,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[MultivariateGaussian, Gaussian, None]:
    """Structural marginalization for node objects.

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
        return Gaussian(
            Scope(marg_scope),
            node.mean[marg_scope_ids[0]],
            np.sqrt(node.cov[marg_scope_ids[0]][marg_scope_ids[0]]),
        )
    # entire node is marginalized over
    elif len(marg_scope) == 0:
        return None
    # node is partially marginalized over
    else:
        # compute marginalized mean vector and covariance matrix
        marg_scope_ids = [scope.query.index(rv) for rv in marg_scope]
        marg_mean = node.mean[marg_scope_ids]
        marg_cov = node.cov[marg_scope_ids][:, marg_scope_ids]

        return MultivariateGaussian(Scope(marg_scope), marg_mean, marg_cov)
