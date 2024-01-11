from typing import List, Tuple

from spflow.meta.data import FeatureContext, FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.meta.dispatch.sampling_context import init_default_sampling_context
from spflow.modules.node.leaf_node import LeafNode
from spflow.modules.node.leaf.utils import apply_nan_strategy
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded

from typing import Callable, Optional, Union

import torch
from torch import nn, Tensor


from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class Gaussian(LeafNode):
    def __init__(self, scope: Scope, mean: float = 0.0, std: float = 1.0) -> None:
        r"""Initializes ``Gaussian`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            mean:
                Floating point value representing the mean (:math:`\mu`) of the distribution.
                Defaults to 0.0.
            std:
                Floating point values representing the standard deviation (:math:`\sigma`) of the distribution (must be greater than 0).
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Gaussian' should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Gaussian' should be empty, but was {scope.evidence}.")
        if std <= 0.0:
            raise ValueError(f"Value for 'std' must be greater than 0.0, but was: {std}")

        super().__init__(scope=scope)

        self.mean = nn.Parameter(torch.tensor(mean))
        self.log_std = nn.Parameter(torch.empty(1))
        self.std = torch.tensor(std)

    @property
    def std(self) -> Tensor:
        """Returns the standard deviation."""
        return self.log_std.exp()

    @std.setter
    def std(self, std) -> Tensor:
        """Set the standard deviation."""
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(std):
            raise ValueError(f"Values for 'std' must be finite, but was: {std}")

        log_std = std.log()
        self.log_std.data = log_std

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Gaussian`` can represent a single univariate node with ``MetaType.Continuous`` or ``GaussianType`` domain.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        feature_ctx = signatures[0]
        domains = feature_ctx.get_domains()

        # leaf is a single non-conditional univariate node
        if (
            len(domains) != 1
            or len(feature_ctx.scope.query) != len(domains)
            or len(feature_ctx.scope.evidence) != 0
        ):
            return False

        # leaf is a continuous Gaussian distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.Gaussian
            or isinstance(domains[0], FeatureTypes.Gaussian)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Gaussian":
        """Creates an instance from a specified signature.

        Returns:
            ``Gaussian`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'Gaussian' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Continuous:
            mean, std = 0.0, 1.0
        elif domain == FeatureTypes.Gaussian:
            # instantiate object
            domain = domain()
            mean, std = domain.mean, domain.std
        elif isinstance(domain, FeatureTypes.Gaussian):
            mean, std = domain.mean, domain.std
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Gaussian' that was not caught during acception checking."
            )

        return Gaussian(feature_ctx.scope, mean=mean, std=std)

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Normal(self.mean, self.std)

    def describe_node(self) -> str:
        return f"mean={self.mean.item():.3f}, std={self.std.item():.3f}"


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: Gaussian,
    data: Tensor,
    weights: Optional[Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``Gaussian`` node parameters in the ``torch`` backend.

    Estimates the mean and standard deviation :math:`\mu` and :math:`\sigma` of a Gaussian distribution from data, as follows:

    .. math::

        \mu^{\*}=\frac{1}{n\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_ix_i\\
        \sigma^{\*}=\frac{1}{\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_i(x_i-\mu^{\*})^2

    or

    .. math::

        \sigma^{\*}=\frac{1}{(\sum_{i=1}^N w_i)-1}\sum_{i=1}^{N}w_i(x_i-\mu^{\*})^2

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

    # apply NaN strategy
    scope_data, weights = apply_nan_strategy(nan_strategy, scope_data, leaf, weights, check_support)

    # normalize weights to sum to n_samples
    weights /= weights.sum() / scope_data.shape[0]

    # total (weighted) number of instances
    n_total = weights.sum()

    # calculate mean and standard deviation from data
    mean_est = (weights * scope_data).sum() / n_total
    std_est = (weights * (scope_data - mean_est) ** 2).sum()

    if bias_correction:
        std_est = torch.sqrt((weights * torch.pow(scope_data - mean_est, 2)).sum() / (n_total - 1))
    else:
        std_est = torch.sqrt((weights * torch.pow(scope_data - mean_est, 2)).sum() / n_total)

    # edge case (if all values are the same, not enough samples or very close to each other)
    if torch.isclose(std_est, torch.tensor(0.0)) or torch.isnan(std_est):
        std_est = torch.tensor(1e-8)

    # set parameters of leaf node
    leaf.mean.data = mean_est
    leaf.log_std.data = std_est.log()
