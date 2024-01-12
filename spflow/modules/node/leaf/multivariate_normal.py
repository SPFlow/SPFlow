"""Contains Multivariate Normal leaf node for SPFlow in the ``torch`` backend.
"""
from spflow.modules.node.leaf import Normal
from spflow.modules.node.leaf_node import LeafNode
import warnings
from collections.abc import Iterable
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from torch import Tensor

from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)
from spflow.utils.nearest_sym_pd import nearest_sym_pd
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded
from typing import Optional

import torch
import torch.distributions as D

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class MultivariateNormal(LeafNode):
    r"""Multivariate Normal distribution leaf node in the 'base' backend.

    Represents a multivariate Normal distribution, with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{(2\pi)^d\det\Sigma}}\exp\left(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)\right)

    where
        - :math:`d` is the dimension of the distribution
        - :math:`x` is the :math:`d`-dim. vector of observations
        - :math:`\mu` is the :math:`d`-dim. mean vector
        - :math:`\Sigma` is the :math:`d\times d` covariance matrix

    Note, that different to ``MultivariateNormal`` in the ``base`` backend, the ``torch`` implementation only accepts positive definite (as opposed to positive semi-definite) covariance matrices.
    Internally :math:`\Sigma` is represented using two unbounded parametes that are combined and projected to yield a positive definite matrix, representing the actual covariance matrix.

    Attributes:
        mean:
            A two-dimensional PyTorch tensor containing the means (:math:`\mu`) of each of the one-dimensional Normal distributions.
            Has exactly as many elements as the scope of this leaf.
        tril_diag_aux:
            Unbounded one-dimensional PyTorch tensor containing values that are projected onto the range :math:`(0,\infty)` for the diagonal entries of the lower triangular Cholesky decomposition of the actual covariance matrix.
        tril_nondiag:
            Unbounded one-dimensional PyTorch tensor containing the values of the non-diagonal elements of the lower triangular Cholesky decomposition of the actual covariance matrix
        cov:
            Two-dimensional PyTorch tensor (representing a :math:`d\times d` symmetric positive definite matrix, where :math:`d` is the length
            of the scope) describing the covariances of the distribution. The diagonal holds the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
            Projected from ``tril_diag_aux`` and ``tril_nondiag``.
    """

    def __init__(
        self,
        scope: Scope,
        mean: Optional[Union[list[float], Tensor, np.ndarray]] = None,
        cov: Optional[Union[list[list[float]], Tensor, np.ndarray]] = None,
    ) -> None:
        r"""Initializes ``MultivariateNormal`` leaf node.

        Args:
            mean:
                A list of floating points, one-dimensional NumPy array or one-dimensional PyTorch tensor containing the means (:math:`\mu`) of each of the one-dimensional Normal distributions.
                Must have exactly as many elements as the scope of this leaf.
                Defaults to all zeros.
            cov:
                A list of lists of floating points, a two-dimensional NumPy array or a two-dimensional PyTorch array (representing a :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
                of the scope) describing the covariances of the distribution. The diagonal holds the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
                Defaults to the identity matrix.
        """
        super().__init__(scope=scope)

        # check if scope contains duplicates
        if len(set(scope.query)) != len(scope.query):
            raise ValueError("Query scope for 'MultivariateNormal' contains duplicate variables.")
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'MultivariateNormal' should be empty, but was {scope.evidence}."
            )
        if len(scope.query) < 1:
            raise ValueError("Size of query scope for 'MultivariateNormal' must be at least 1.")

        if mean is None:
            mean = torch.zeros((1, len(scope)))
        if cov is None:
            cov = torch.eye(len(scope))

        # dimensions
        self.d = len(scope)

        # register mean vector as torch parameters
        self.mean = Parameter()

        # internally we use the lower triangular matrix (Cholesky decomposition) to encode the covariance matrix
        # register (auxiliary) values for diagonal and non-diagonal values of lower triangular matrix as torch parameters
        self.tril_diag_aux = Parameter()
        self.tril_nondiag = Parameter()

        # pre-compute and store indices of non-diagonal values for lower triangular matrix
        self.tril_nondiag_indices = torch.tril_indices(self.d, self.d, offset=-1)

        # set parameters
        self.set_params(mean, cov)
        self.backend = "pytorch"

    @property
    def covariance_tril(self) -> Tensor:
        """Returns the lower triangular matrix of the Cholesky-decomposed covariance matrix."""
        # create zero matrix of appropriate dimension
        L_nondiag = torch.zeros((self.d, self.d), device=self.device)
        # fill non-diagonal values of lower triangular matrix
        L_nondiag[self.tril_nondiag_indices[0], self.tril_nondiag_indices[1]] = self.tril_nondiag
        # add (projected) diagonal values
        L = L_nondiag + proj_real_to_bounded(self.tril_diag_aux, lb=0.0) * torch.eye(
            self.d, device=self.device
        )
        # return lower triangular matrix
        return L

    @property
    def cov(self) -> Tensor:
        """Returns the covariance matrix."""
        # get lower triangular matrix
        L = self.covariance_tril
        # return covariance matrix
        return torch.matmul(L, L.T)

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``MultivariateNormal`` can represent a single univariate node with ``MetaType.Continuous`` or ``NormalType`` domains.

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

        # leaf is a continuous (multivariate) Normal distribution
        if not all(
            [
                domain == FeatureTypes.Continuous
                or domain == FeatureTypes.Normal
                or isinstance(domain, FeatureTypes.Normal)
                for domain in domains
            ]
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "MultivariateNormal":
        """Creates an instance from a specified signature.

        Returns:
            ``MultivariateNormal`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'MultivariateNormal' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]

        mean, cov = np.zeros(len(feature_ctx.scope.query)), np.eye(len(feature_ctx.scope.query))

        for i, domain in enumerate(feature_ctx.get_domains()):
            # read or initialize parameters
            if domain == MetaType.Continuous:
                pass
            elif domain == FeatureTypes.Normal:
                # instantiate object
                domain = domain()
                mean[i], cov[i][i] = domain.mean, domain.std
            elif isinstance(domain, FeatureTypes.Normal):
                mean[i], cov[i][i] = domain.mean, domain.std
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'MultivariateNormal' that was not caught during acception checking."
                )

        return MultivariateNormal(feature_ctx.scope, mean=mean, cov=cov)

    @property
    def distribution(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.MultivariateNormal`` instance.
        """
        return D.MultivariateNormal(loc=self.mean, scale_tril=self.covariance_tril)

    def set_params(
        self,
        mean: Union[list[float], Tensor, np.ndarray],
        cov: Union[list[list[float]], Tensor, np.ndarray],
    ) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            mean:
                A list of floating points, one-dimensional NumPy array or one-dimensional PyTorch tensor containing the means (:math:`\mu`) of each of the one-dimensional Normal distributions.
                Must have exactly as many elements as the scope of this leaf.
                Defaults to all zeros.
            cov:
                A list of lists of floating points, a two-dimensional NumPy array or a two-dimensional PyTorch array (representing a :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
                of the scope) describing the covariances of the distribution. The diagonal holds the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
                If a positive semi-definite matrix is specified, the closest positive definite matrix in the Frobenius norm is used instead.
                Defaults to the identity matrix.
        """
        if isinstance(mean, list):
            # convert float list to torch tensor
            mean = torch.tensor([float(v) for v in mean], device=self.device)
        elif isinstance(mean, np.ndarray):
            # convert numpy array to torch tensor
            mean = torch.from_numpy(mean).to(self.device)

        if isinstance(cov, list):
            # convert numpy array to torch tensor
            cov = torch.tensor([[float(v) for v in row] for row in cov], device=self.device)
        elif isinstance(cov, np.ndarray):
            # convert numpy array to torch tensor
            cov = torch.from_numpy(cov).to(self.device)

        # check mean vector for nan or inf values
        if torch.any(torch.isinf(mean)):
            raise ValueError("Value of 'mean' for 'MultivariateNormal' may not contain infinite values")
        if torch.any(torch.isnan(mean)):
            raise ValueError("Value of 'mean' for 'MultivariateNormal' may not contain NaN values")

        # make sure that number of dimensions matches scope length
        if (
            (mean.ndim == 1 and mean.shape[0] != len(self.scope.query))
            or (mean.ndim == 2 and mean.shape[1] != len(self.scope.query))
            or mean.ndim > 2
        ):
            raise ValueError(
                f"Dimensions of 'mean' for 'MultivariateNormal' should match scope size {len(self.scope.query)}, but was: {mean.shape}"
            )

        if mean.ndim == 2:
            mean = mean.squeeze(0)

        # make sure that dimensions of covariance matrix are correct
        if cov.ndim != 2 or (
            cov.ndim == 2 and (cov.shape[0] != len(self.scope.query) or cov.shape[1] != len(self.scope.query))
        ):
            raise ValueError(
                f"Value of 'cov' for 'MultivariateNormal' expected to be of shape ({len(self.scope.query), len(self.scope.query)}), but was: {cov.shape}"
            )

        # set mean vector
        self.mean.data = mean.to(self.device)

        # check covariance matrix for nan or inf values
        if torch.any(torch.isinf(cov)):
            raise ValueError("Value of 'cov' for 'MultivariateNormal' may not contain infinite values")
        if torch.any(torch.isnan(cov)):
            raise ValueError("Value of 'cov' for 'MultivariateNormal' may not contain NaN values")

        # compute eigenvalues (can use eigvalsh here since we already know matrix is symmetric)
        eigvals = torch.linalg.eigvalsh(cov)

        if torch.any(eigvals < 0.0):
            raise ValueError(
                "Value of 'cov' for 'MultivariateNormal' is not symmetric positive semi-definite (contains negative real eigenvalues)."
            )

        # edge case: covariance matrix is positive semi-definite but NOT positive definite (needed for projection)
        if torch.any(eigvals == 0):
            warnings.warn(
                "Value of 'cov' for 'MultivariateNormal' is positive semi-definite, but not positive definite. Using closest positive definite matrix instead. Required for interal projection of learnable parameters.",
                RuntimeWarning,
            )
            # find nearest symmetric positive definite matrix in Frobenius norm
            cov = nearest_sym_pd(cov)

        # compute lower triangular matrix
        L = torch.linalg.cholesky(cov).to(self.device)  # type: ignore

        # set diagonal and non-diagonal values of lower triangular matrix
        self.tril_diag_aux.data = proj_bounded_to_real(torch.diag(L), lb=0.0)
        self.tril_nondiag.data = L[self.tril_nondiag_indices[0], self.tril_nondiag_indices[1]]

    def check_support(self, data: Tensor, is_scope_data: bool = False) -> Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Multivariate Normal distribution, which is:

        .. math::

            \text{supp}(\text{MultivariateNormal})=(-\infty,+\infty)^k

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leaf's scope in the correct order (True) or if it needs to be extracted from the full data set.
                Defaults to False.

        Returns:
            Two-dimensional PyTorch tensor indicating for each instance, whether they are part of the support (True) or not (False).
        """
        if is_scope_data:
            scope_data = data
        else:
            # select relevant data for scope
            scope_data = data[:, self.scope.query]

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scopes_out[0].query):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scopes_out[0].query)}), but was: {scope_data.shape}"
            )

        # different to univariate distributions, cannot simply check via torch distribution's support due to possible incomplete data in multivariate case; therefore do it ourselves (not difficult here since support is R)
        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool, device=self.device)

        # check for infinite values (may return NaNs despite support)
        valid &= ~scope_data.isinf().sum(dim=1, keepdim=True).bool()

        return valid


@dispatch(memoize=True)  # type: ignore
def marginalize(
    node: MultivariateNormal,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[MultivariateNormal, Normal, None]:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # scope after marginalization (important: must remain order of scope indices since they map to the indices of the mean vector and covariance matrix!)
    marg_scope = []
    marg_scope_ids = []

    scope = node.scope

    for rv in scope.query:
        if rv not in marg_rvs:
            marg_scope.append(rv)
            marg_scope_ids.append(scope.query.index(rv))

    # return univariate Normal if one-dimensional
    if len(marg_scope) == 1:
        # note: Normal requires standard deviations instead of variance (take square root)
        return Normal(
            Scope(marg_scope),
            node.mean[marg_scope_ids[0]].detach().cpu().item(),
            torch.sqrt(node.cov[marg_scope_ids[0]][marg_scope_ids[0]].detach()).cpu().item(),
        )
    # entire node is marginalized over
    elif len(marg_scope) == 0:
        return None
    # node is partially marginalized over
    else:
        # compute marginalized mean vector and covariance matrix
        marg_mean = node.mean[marg_scope_ids]
        marg_cov = node.cov[marg_scope_ids][:, marg_scope_ids]

        return MultivariateNormal(Scope(marg_scope), marg_mean, marg_cov)


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: MultivariateNormal,
    data: Tensor,
    weights: Optional[Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``MultivariateNormal`` node parameters in the ``torch`` backend.

    Estimates the mean and covariance matrix :math:`\mu` and :math:`\Sigma` of a Multivariate Normal distribution from data, as follows:

    .. math::

        \mu^{\*}=\frac{1}{n\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_ix_i\\
        \Sigma^{\*}=\frac{1}{\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_i(x_i-\mu^{\*})(x_i-\mu^{\*})^T

    or

    .. math::

        \Sigma^{\*}=\frac{1}{(\sum_{i=1}^N w_i)-1}\sum_{i=1}^{N}w_i(x_i-\mu^{\*})(x_i-\mu^{\*})^T

    if bias correction is used, where
        - :math:`N` is the number of samples in the data set
        - :math:`x_i` is the data of the relevant scope for the `i`-th sample of the data set
        - :math:`w_i` is the weight for the `i`-th sample of the data set

    Weights are normalized to sum up to :math:`N`.

    Args:
        leaf:
            Leaf node to estimate parameters of.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        weights:
            Optional one-dimensional PyTorch tensor containing non-negative weights for all data samples.
            Must match number of samples in ``data``.
            Defaults to None in which case all weights are initialized to ones.
        bias_corrections:
            Boolen indicating whether or not to correct possible biases.
            Defaults to True.
        nan_strategy:
            Optional string or callable specifying how to handle missing data.
            If 'ignore', missing values (i.e., NaN entries) are ignored.
            If a callable, it is called using ``data`` and should return another PyTorch tensor of same size.
            Defaults to None.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Raises:
        ValueError: Invalid arguments.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    if weights is None:
        weights = torch.ones(data.shape[0], device=data.device)

    if weights.ndim != 1 or weights.shape[0] != data.shape[0]:
        raise ValueError(
            "Number of specified weights for maximum-likelihood estimation does not match number of data points."
        )

    # reshape weights
    weights = weights.reshape(-1, 1)

    if check_support:
        if torch.any(~leaf.check_support(scope_data, is_scope_data=True)):
            raise ValueError("Encountered values outside of the support for 'MultivariateNormal'.")

    # NaN entries (no information)
    nan_mask = torch.isnan(scope_data)

    if torch.all(nan_mask):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data.")

    if nan_strategy is None and torch.any(nan_mask):
        raise ValueError(
            "Maximum-likelihood estimation cannot be performed on missing data by default. Set a strategy for handling missing values if this is intended."
        )

    if isinstance(nan_strategy, str):
        if nan_strategy == "ignore":
            pass  # handle it during computation
        else:
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'MultivariateNormal'.")
    elif isinstance(nan_strategy, Callable):
        scope_data = nan_strategy(scope_data)
        # TODO: how to handle weights?
    elif nan_strategy is not None:
        raise ValueError(
            f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}."
        )

    # normalize weights to sum to n_samples
    weights /= weights.sum() / scope_data.shape[0]

    if nan_strategy == "ignore":
        n_total = (weights * ~nan_mask).sum(dim=0)
        # compute mean of available data
        mean_est = torch.sum(weights * torch.nan_to_num(scope_data, nan=0.0), dim=0) / n_total
        # compute covariance of full samples only!
        full_sample_mask = (~nan_mask).sum(dim=1) == scope_data.shape[1]
        cov_est = torch.cov(
            scope_data[full_sample_mask].T,
            aweights=weights[full_sample_mask].squeeze(-1),
            correction=1 if bias_correction else 0,
        )
    else:
        n_total = (weights * ~nan_mask).sum(dim=0)
        # calculate mean and standard deviation from data
        mean_est = (weights * scope_data).sum(dim=0) / n_total
        cov_est = torch.cov(
            scope_data.T,
            aweights=weights.squeeze(-1),
            correction=1 if bias_correction else 0,
        )

    if len(leaf.scope.query) == 1:
        cov_est = cov_est.reshape(1, 1)

    # edge case (if all values are the same, not enough samples or very close to each other)
    for i in range(scope_data.shape[1]):
        if torch.isclose(cov_est[i][i], torch.tensor(0.0)):
            cov_est[i][i] = 1e-8

    # sometimes numpy returns a matrix with non-positive eigenvalues (i.e., not a valid positive definite matrix)
    # NOTE: we need test for non-positive here instead of negative for NumPy, because we need to be able to perform cholesky decomposition
    if torch.any(torch.linalg.eigvalsh(cov_est) <= 0):
        # compute nearest symmetric positive semidefinite matrix
        cov_est = nearest_sym_pd(cov_est)

    # set parameters of leaf node
    leaf.set_params(mean=mean_est, cov=cov_est)


@dispatch  # type: ignore
def sample(
    leaf: MultivariateNormal,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    r"""Samples from ``MultivariateNormal`` nodes in the ``torch`` backend given potential evidence.

    Samples missing values proportionally to its probability distribution function (PDF).
    If evidence is present, values are sampled from the conitioned marginal distribution.

    Args:
        leaf:
            Leaf node to sample from.
        data:
            Two-dimensional PyTorch tensor containing potential evidence.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
        sampling_ctx:
            Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

    Returns:
        Two-dimensional PyTorch tensor containing the sampled values together with the specified evidence.
        Each row corresponds to a sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    if any([i >= data.shape[0] for i in sampling_ctx.instance_ids]):
        raise ValueError("Some instance ids are out of bounds for data tensor.")

    # compute nan_mask for specified instances
    instances_mask = torch.zeros(data.shape[0], device=leaf.device).bool()
    instances_mask[torch.tensor(sampling_ctx.instance_ids)] = True

    nan_data = torch.isnan(
        data[
            torch.meshgrid(
                torch.where(instances_mask)[0],
                torch.tensor(leaf.scope.query, device=leaf.device),
                indexing="ij",
            )
        ]
    )

    # group by scope rvs to sample
    for nan_mask in torch.unique(nan_data, dim=0):
        cond_rvs = torch.tensor(leaf.scope.query, device=leaf.device)[
            torch.where(~nan_mask)[0]
        ]  # ids for evidence RVs
        non_cond_rvs = torch.tensor(leaf.scope.query, device=leaf.device)[
            torch.where(nan_mask)[0]
        ]  # RVs to be sampled

        # no 'NaN' values (nothing to sample)
        if torch.sum(nan_mask) == 0:
            continue
        # sample from full distribution
        elif torch.sum(nan_mask) == len(leaf.scope.query):
            sampling_ids = torch.tensor(sampling_ctx.instance_ids, device=leaf.device)[
                (nan_data == nan_mask).sum(dim=1) == nan_mask.shape[0]
            ]

            data[torch.meshgrid(sampling_ids, non_cond_rvs, indexing="ij")] = leaf.distribution.sample(
                (sampling_ids.shape[0],)
            ).squeeze(1)
        # sample from conditioned distribution
        else:
            # note: the conditional sampling implemented here is based on the algorithm described in Arnaud Doucet (2010): "A Note on Efficient Conditional Simulation of Normal Distributions" (https://www.stats.ox.ac.uk/~doucet/doucet_simulationconditionalgaussian.pdf)
            sampling_ids = torch.tensor(sampling_ctx.instance_ids, device=leaf.device)[
                (nan_data == nan_mask).sum(dim=1) == nan_mask.shape[0]
            ]

            # sample from full distribution
            joint_samples = leaf.distribution.sample((sampling_ids.shape[0],))

            # compute inverse of marginal covariance matrix of conditioning RVs
            marg_cov_inv = torch.linalg.inv(leaf.cov[torch.meshgrid(cond_rvs, cond_rvs, indexing="ij")])

            # get conditional covariance matrix
            cond_cov = leaf.cov[torch.meshgrid(cond_rvs, non_cond_rvs, indexing="ij")]

            data[torch.meshgrid(sampling_ids, non_cond_rvs, indexing="ij")] = joint_samples[:, nan_mask] + (
                (data[torch.meshgrid(sampling_ids, cond_rvs, indexing="ij")] - joint_samples[:, ~nan_mask])
                @ (marg_cov_inv @ cond_cov)
            )

    return data


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    leaf: MultivariateNormal,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    r"""Computes log-likelihoods for ``MultivariateNormal`` node in the ``torch`` backend given input data.

    Log-likelihood for ``MultivariateNormal`` is given by the logarithm of its probability distribution function (PDF):

    .. math::

        \log(\text{PDF}(x)) = \log(\frac{1}{\sqrt{(2\pi)^d\det\Sigma}}\exp\left(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)\right))

    where
        - :math:`d` is the dimension of the distribution
        - :math:`x` is the :math:`d`-dim. vector of observations
        - :math:`\mu` is the :math:`d`-dim. mean vector
        - :math:`\Sigma` is the :math:`d\times d` covariance matrix

    Missing values (i.e., NaN) are marginalized over.

    Args:
        node:
            Leaf node to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the distribution.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.

    Raises:
        ValueError: Data outside of support.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    batch_size: int = data.shape[0]

    # get information relevant for the scope
    scope_data = data[:, leaf.scope.query]

    # initialize empty tensor (number of output values matches batch_size)
    log_prob = torch.empty(batch_size, 1, dtype=torch.float, device=data.device)

    # create copy of the data where NaNs are replaced by zeros
    # TODO: alternative for initial validity checking without copying?
    _scope_data = scope_data.clone()
    _scope_data[_scope_data.isnan()] = 0.0

    if check_support:
        # check support
        valid_ids = leaf.check_support(_scope_data, is_scope_data=True).squeeze(1)

        if not all(valid_ids):
            raise ValueError(
                f"Encountered data instances that are not in the support of the TorchMultivariateNormal distribution."
            )

    del _scope_data  # free up memory

    # ----- log probabilities -----

    marg = torch.isnan(scope_data)

    # group instances by marginalized random variables
    for marg_mask in marg.unique(dim=0):
        # get all instances with the same (marginalized) scope
        marg_ids = torch.where((marg == marg_mask).sum(dim=-1) == len(leaf.scope.query))[0]
        marg_data = scope_data[marg_ids]

        # all random variables are marginalized over
        if all(marg_mask):
            # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
            log_prob[marg_ids, 0] = 0.0
        # some random variables are marginalized over
        elif any(marg_mask):
            marg_data = marg_data[:, ~marg_mask]

            # marginalize distribution and compute (log) probabilities
            marg_mean = leaf.mean[~marg_mask]
            marg_cov = leaf.cov[~marg_mask][:, ~marg_mask]  # TODO: better way?

            # create marginalized torch distribution
            marg_dist = D.MultivariateNormal(loc=marg_mean, covariance_matrix=marg_cov)

            # compute probabilities for values inside distribution support
            log_prob[marg_ids, 0] = marg_dist.log_prob(marg_data)
        # no random variables are marginalized over
        else:
            # compute probabilities for values inside distribution support
            log_prob[marg_ids, 0] = leaf.distribution.log_prob(marg_data)

    return log_prob
