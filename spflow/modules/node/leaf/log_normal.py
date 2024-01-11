"""Contains Log-Normal leaf node for SPFlow in the ``torch`` backend.
"""
from spflow.modules.node.leaf.utils import apply_nan_strategy
from spflow.modules.node.leaf_node import LeafNode
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D

from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded
from typing import Callable, Optional, Union

import torch
from torch import nn, Tensor

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class LogNormal(LeafNode):
    r"""(Univariate) Log-Normal distribution leaf node in the ``torch`` backend.

    Represents an univariate Log-Normal distribution, with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln(x)-\mu)^2}{2\sigma^2}\right)

    where
        - :math:`x` is an observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Internally :math:`\mu,\sigma` are represented as unbounded parameters that are projected onto the bounded range :math:`(0,\infty)` for representing the actual shape and rate parameters, respectively.

    Attributes:
        mean:
            Scalar PyTorch tensor representing the mean (:math:`\mu`) of the Gamma distribution.
        std_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual standard deviation.
        std:
            Scalar PyTorch tensor representing the standard deviation (:math:`\sigma`) of the Gaussian distribution, greater than 0 (projected from ``std_aux``).
    """

    def __init__(
        self,
        scope: Scope,
        mean: Optional[float] = 0.0,
        std: Optional[float] = 1.0,
    ) -> None:
        r"""Initializes ``LogNormal`` leaf node.

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
            raise ValueError(f"Query scope size for 'LogNormal' should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'LogNormal' should be empty, but was {scope.evidence}.")

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

        if std <= 0.0:
            raise ValueError(f"Value for 'std' must be greater than 0.0, but was: {std}")

        log_std = std.log()
        self.log_std.data = log_std

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``LogNormal`` can represent a single univariate node with ``MetaType.Continuous`` or ``LogNormalType`` domain.

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

        # leaf is a continuous Log-Normal distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.LogNormal
            or isinstance(domains[0], FeatureTypes.LogNormal)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "LogNormal":
        """Creates an instance from a specified signature.

        Returns:
            ``LogNormal`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'LogNormal' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Continuous:
            mean, std = 0.0, 1.0
        elif domain == FeatureTypes.LogNormal:
            # instantiate object
            domain = domain()
            mean, std = domain.mean, domain.std
        elif isinstance(domain, FeatureTypes.LogNormal):
            mean, std = domain.mean, domain.std
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'LogNormal' that was not caught during acception checking."
            )

        return LogNormal(feature_ctx.scope, mean=mean, std=std)

    @property
    def distribution(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.LogNormal`` instance.
        """
        return D.LogNormal(loc=self.mean, scale=self.std)


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: LogNormal,
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``LogNormal`` node parameters in the ``torch`` backend.

    Estimates the mean and standard deviation :math:`\mu` and :math:`\sigma` of a Log-Normal distribution from data, as follows:

    .. math::

        \mu^{\*}=\frac{1}{n\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_i\log(x_i)\\
        \sigma^{\*}=\frac{1}{\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_i(\log(x_i)-\mu^{\*})^2

    or

    .. math::

        \sigma^{\*}=\frac{1}{(\sum_{i=1}^N w_i)-1}\sum_{i=1}^{N}w_i(\log(x_i)-\mu^{\*})^2

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

    # Apply NaN strategy
    scope_data, weights = apply_nan_strategy(nan_strategy, scope_data, leaf, weights, check_support)

    # total (weighted) number of instances
    n_total = weights.sum()

    # calculate mean and standard deviation from data
    mean_est = (weights * torch.log(scope_data)).sum() / n_total

    if bias_correction:
        std_est = torch.sqrt((weights * torch.pow(torch.log(scope_data) - mean_est, 2)).sum() / (n_total - 1))
    else:
        std_est = torch.sqrt((weights * torch.pow(torch.log(scope_data) - mean_est, 2)).sum() / n_total)

    # edge case (if all values are the same, not enough samples or very close to each other)
    if torch.isclose(std_est, torch.tensor(0.0)) or torch.isnan(std_est):
        std_est = torch.tensor(1e-8)

    # set parameters of leaf node
    leaf.mean.data = mean_est
    leaf.log_std.data = std_est.log()
