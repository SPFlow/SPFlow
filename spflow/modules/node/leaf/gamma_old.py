"""Contains Gamma leaf node for SPFlow in the ``torch`` backend.
"""
from spflow.modules.node.leaf.utils import apply_nan_strategy
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded
from spflow.modules.node.leaf_node import LeafNode
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter

from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from typing import Callable, Optional, Union

import torch
from torch import nn, Tensor

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class Gamma(LeafNode):
    r"""(Univariate) Gamma distribution leaf node in the ``base`` backend.

    Represents an univariate Exponential distribution, with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \begin{cases} \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x} & \text{if } x > 0\\
                                      0 & \text{if } x <= 0\end{cases}

    where
        - :math:`x` is the input observation
        - :math:`\Gamma` is the Gamma function
        - :math:`\alpha` is the shape parameter
        - :math:`\beta` is the rate parameter

    Internally :math:`\alpha,\beta` are represented as unbounded parameters that are projected onto the bounded range :math:`(0,\infty)` for representing the actual shape and rate parameters, respectively.

    Attributes:
        alpha_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual shape parameter.
        alpha:
            Scalar PyTorch tensor representing the shape parameter (:math:`\alpha`) of the Gamma distribution, greater than 0 (projected from ``alpha_aux``).
        beta_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual rate parameter.
        beta:
            Scalar PyTorch tensor representing the rate parameter (:math:`\beta`) of the Gamma distribution, greater than 0 (projected from ``beta_aux``).
    """

    def __init__(self, scope: Scope, alpha: float = 1.0, beta: float = 1.0) -> None:
        r"""Initializes ``Exponential`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            alpha:
                Floating point value representing the shape parameter (:math:`\alpha`), greater than 0.
                Defaults to 1.0.
            beta:
                Floating point value representing the rate parameter (:math:`\beta`), greater than 0.
                Defaults to 1.0.
        """
        super().__init__(scope=scope)
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Gamma' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Gamma' should be empty, but was {scope.evidence}.")

        # register auxiliary torch parameters for alpha and beta
        self.alpha_aux = nn.Parameter(torch.empty(1))
        self.alpha = torch.tensor(alpha)
        self.beta_aux = nn.Parameter(torch.empty(1))
        self.beta = torch.tensor(beta)

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Gamma`` can represent a single univariate node with ``MetaType.Continuous`` or ``GammaType`` domain.

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

        # leaf is a continuous Gamma distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.Gamma
            or isinstance(domains[0], FeatureTypes.Gamma)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Gamma":
        """Creates an instance from a specified signature.

        Returns:
            ``Gamma`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'Gamma' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Continuous:
            alpha, beta = 1.0, 1.0
        elif domain == FeatureTypes.Gamma:
            # instantiate object
            domain = domain()
            alpha, beta = domain.alpha, domain.beta
        elif isinstance(domain, FeatureTypes.Gamma):
            alpha, beta = domain.alpha, domain.beta
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Gamma' that was not caught during acception checking."
            )

        return Gamma(feature_ctx.scope, alpha=alpha, beta=beta)

    @property
    def distribution(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.Gamma`` instance.
        """
        return D.Gamma(concentration=self.alpha, rate=self.beta)

    @property
    def alpha(self) -> torch.Tensor:
        """Returns the shape parameter."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.alpha_aux, lb=0.0)  # type: ignore

    @property
    def beta(self) -> torch.Tensor:
        """Returns the rate parameter."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.beta_aux, lb=0.0)  # type: ignore

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError(
                f"Value of alpha for Gamma distribution must be greater than 0, but was: {alpha}"
            )
        self.alpha_aux.data = proj_bounded_to_real(alpha, lb=0.0)

    @beta.setter
    def beta(self, beta: torch.Tensor) -> None:
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError(f"Value of beta for Gamma distribution must be greater than 0, but was: {beta}")
        self.beta_aux.data = proj_bounded_to_real(beta, lb=0.0)


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: Gamma,
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``Gamma`` node parameters in the ``torch`` backend.

    Estimates the shape and rate parameters :math:`alpha`,:math:`beta` of a Gamma distribution from data, as described in (Minka, 2002): "Estimating a Gamma distribution" (adjusted to support weights).
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
            Has no effect for ``Gamma`` nodes.
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

    # compute two parameter gamma estimates according to (Mink, 2002): https://tminka.github.io/papers/minka-gamma.pdf
    # also see this VBA implementation for reference: https://github.com/jb262/MaximumLikelihoodGammaDist/blob/main/MLGamma.bas
    # adapted to take weights

    n_total = weights.sum()
    mean = (weights * scope_data).sum() / n_total
    log_mean = mean.log()
    mean_log = (weights * scope_data.log()).sum() / n_total

    # start values
    alpha_prev = torch.tensor(0.0)
    alpha_est = 0.5 / (log_mean - mean_log)

    # iteratively compute alpha estimate
    while torch.abs(alpha_prev - alpha_est) > 1e-6:
        alpha_prev = alpha_est
        alpha_est = 1.0 / (
            1.0 / alpha_prev
            + (mean_log - log_mean + alpha_prev.log() - torch.digamma(alpha_prev))
            / (alpha_prev**2 * (1.0 / alpha_prev - torch.polygamma(n=1, input=alpha_prev)))
        )

    # compute beta estimate
    # NOTE: different to the original paper we compute the inverse since beta=1.0/scale
    beta_est = alpha_est / mean

    # TODO: bias correction?

    # edge case: if alpha/beta 0, set to larger value (should not happen, but just in case)
    if torch.isclose(alpha_est, torch.tensor(0.0)):
        alpha_est = torch.tensor(1e-8)
    if torch.isclose(beta_est, torch.tensor(0.0)):
        beta_est = torch.tensor(1e-8)

    # set parameters of leaf node
    leaf.alpha = alpha_est
    leaf.beta = beta_est
