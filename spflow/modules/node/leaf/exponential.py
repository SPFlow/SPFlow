"""Contains Exponential leaf node for SPFlow in the ``torch`` backend.
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


class Exponential(LeafNode):
    r"""(Univariate) Exponential distribution leaf node in the ``torch`` backend.

    Represents an univariate Exponential distribution, with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \begin{cases} \lambda e^{-\lambda x} & \text{if } x > 0\\
                                      0                      & \text{if } x <= 0\end{cases}

    where
        - :math:`x` is the input observation
        - :math:`\lambda` is the rate parameter

    Internally :math:`l` is represented as an unbounded parameter that is projected onto the bounded range :math:`(0,\infty)` for representing the actual rate parameters.

    Attributes:
        l_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual rate parameter.
        l:
            Scalar PyTorch tensor representing the rate parameter (:math:`\lambda`) of the Exponential distribution (projected from ``l_aux``).
    """

    def __init__(self, scope: Scope, rate: float = 1.0) -> None:
        r"""Initializes ``Exponential`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            rate:
                Floating point value representing the rate parameter (:math:`\lambda`) of the Exponential distribution (must be greater than 0).
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Exponential' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Exponential' should be empty, but was {scope.evidence}.")

        super().__init__(scope=scope)

        # register auxiliary torch parameter for lambda l
        self.rate_aux = nn.Parameter(torch.empty(1))
        self.rate = torch.tensor(rate)

    @property
    def rate(self) -> torch.Tensor:
        """Returns the rate parameter."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.rate_aux, lb=0.0)  # type: ignore

    @rate.setter
    def rate(self, value: torch.Tensor) -> None:
        """Sets the rate parameter."""
        if value < 0:
            raise ValueError(f"Value of rate for Poisson distribution must be non-negative, but was: {value}")
        self.rate_aux.data = proj_bounded_to_real(value, lb=0.0)

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Exponential`` can represent a single univariate node with ``MetaType.Continuous`` or ``ExponentialType`` domain.

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

        # leaf is a discrete Exponential distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.Exponential
            or isinstance(domains[0], FeatureTypes.Exponential)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Exponential":
        """Creates an instance from a specified signature.

        Returns:
            ``Exponential`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'Exponential' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Continuous:
            rate = 1.0
        elif domain == FeatureTypes.Exponential:
            # instantiate object
            rate = domain().l
        elif isinstance(domain, FeatureTypes.Exponential):
            rate = domain.l
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Exponential' that was not caught during acception checking."
            )

        return Exponential(feature_ctx.scope, rate=rate)

    @property
    def distribution(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.Exponential`` instance.
        """
        return D.Exponential(rate=self.rate)


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: Exponential,
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``Exponential`` node parameters in the ``torch`` backend.

    Estimates the rate parameter :math:`l` of an Exponential distribution from data, as follows:

    .. math::

        l^{\*}=\frac{\sum_{i=1}^N w_i}{\sum_{i=1}^N w_ix_i}

    or

    .. math::

        l^{\*}=\frac{(\sum_{i=1}^N w_i)-1}{\sum_{i=1}^N w_ix_i}

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
    scope_data, weights = apply_nan_strategy(
        nan_strategy, scope_data, leaf, weights, check_support=check_support
    )

    # normalize weights to sum to n_samples
    weights /= weights.sum() / scope_data.shape[0]

    # total number of instances
    n_total = weights.sum()

    if bias_correction:
        n_total -= 1

    # cummulative evidence
    cum_rate = (weights * scope_data).sum()

    # estimate rate parameter
    rate_est = n_total / cum_rate

    # edge case: if rate 0, set to larger value (should not happen, but just in case)
    if torch.isclose(rate_est, torch.tensor(0.0)):
        rate_est = torch.tensor(1e-8)

    # set parameters of leaf node
    leaf.rate = rate_est
