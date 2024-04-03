#!/usr/bin/env python3

import torch
from torch import Tensor, nn

from spflow.distributions.distribution import Distribution
from spflow.meta.data import FeatureContext, FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.modules.node.leaf.utils import init_parameter


class Gamma(Distribution):
    def __init__(self, alpha: Tensor = None, beta: Tensor = None, event_shape: tuple[int, ...] = None):
        r"""Initializes ``Gamma`` leaf node.

        Args:
            scope: Scope object specifying the scope of the distribution.
            alpha: Tensor representing the shape parameters (:math:`\alpha`) of the Gamma distributions, greater than 0.
            beta: Tensor representing the rate parameters (:math:`\beta`) of the Gamma distributions, greater than 0.
            n_out: Number of nodes per scope. Only relevant if alpha and beta is None.
        """
        if event_shape is None:
            event_shape = alpha.shape
        super().__init__(event_shape=event_shape)
        assert (alpha is None and beta is None) ^ (
            alpha is not None and beta is not None
        ), "Either alpha and beta must be specified or neither."

        alpha = init_parameter(param=alpha, event_shape=event_shape, init=lambda: torch.tensor(1.0))
        beta = init_parameter(param=beta, event_shape=event_shape, init=lambda: torch.tensor(1.0))

        self.log_alpha = nn.Parameter(alpha)
        self.alpha = alpha.clone().detach()
        self.log_beta = nn.Parameter(torch.empty_like(beta))  # initialize empty, set with setter in next line
        self.beta = beta.clone().detach()

    @property
    def alpha(self) -> Tensor:
        """Returns alpha."""
        return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, alpha):
        """Set alpha."""
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(alpha).all():
            raise ValueError(f"Values for 'beta' must be finite, but was: {alpha}")

        if torch.all(alpha <= 0.0):
            raise ValueError(f"Value for 'beta' must be greater than 0.0, but was: {alpha}")

        self.log_alpha.data = alpha.log()

    @property
    def beta(self) -> Tensor:
        """Returns beta."""
        return self.log_beta.exp()

    @beta.setter
    def beta(self, beta):
        """Set beta."""
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(beta).all():
            raise ValueError(f"Values for 'beta' must be finite, but was: {beta}")

        if torch.all(beta <= 0.0):
            raise ValueError(f"Value for 'beta' must be greater than 0.0, but was: {beta}")

        self.log_beta.data = beta.log()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Gamma(self.alpha, self.beta)

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
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

        return Gamma(alpha=alpha, beta=beta)

    def maximum_likelihood_estimation(self, data: Tensor, weights: Tensor = None, bias_correction=True):
        if weights is None:
            _shape = (data.shape[0], *([1] * (data.dim() - 1)))  # (batch, 1, 1, ...) for broadcasting
            weights = torch.ones(_shape, device=data.device)

        # total (weighted) number of instances
        n_total = weights.sum()

        mean = (weights * data).sum(dim=0) / n_total
        log_mean = mean.log()
        mean_log = (weights * data.log()).sum(
            dim=0
        ) / n_total

        # compute two parameter gamma estimates according to (Mink, 2002): https://tminka.github.io/papers/minka-gamma.pdf
        # also see this VBA implementation for reference: https://github.com/jb262/MaximumLikelihoodGammaDist/blob/main/MLGamma.bas

        # mean, log_mean and mean_log already calculated above

        # start values
        alpha_prev = torch.zeros(data.shape[1], device=data.device)
        alpha_est = 0.5 / (log_mean - mean_log)

        # iteratively compute alpha estimate
        while torch.any(torch.abs(alpha_prev - alpha_est) > 1e-6):
            # mask to only further refine relevant estimates
            iter_mask = torch.abs(alpha_prev - alpha_est) > 1e-6
            alpha_prev = alpha_est
            alpha_est[iter_mask] = 1.0 / (
                    1.0 / alpha_prev[iter_mask]
                    + (
                            mean_log[iter_mask]
                            - log_mean[iter_mask]
                            + alpha_prev[iter_mask].log()
                            - torch.digamma(alpha_prev[iter_mask])
                    )
                    / (
                            alpha_prev[iter_mask] ** 2
                            * (1.0 / alpha_prev[iter_mask] - torch.polygamma(n=1, input=alpha_prev[iter_mask]))
                    )
            )

        # compute beta estimate
        # NOTE: different to the original paper we compute the inverse since beta=1.0/scale
        beta_est = alpha_est / mean

        # edge case (if all values are the same, not enough samples or very close to each other)
        if torch.any(zero_mask := torch.isclose(beta_est, torch.tensor(0.0))):
            beta_est[zero_mask] = torch.tensor(1e-8)
        if torch.any(nan_mask := torch.isnan(beta_est)):
            beta_est[nan_mask] = torch.tensor(1e-8)

        if len(self.event_shape) == 2:
            # Repeat alpha and beta
            alpha_est = alpha_est.unsqueeze(1).repeat(1, self.event_shape[1])
            beta_est = beta_est.unsqueeze(1).repeat(1, self.event_shape[1])

        # set parameters of leaf node
        self.alpha.data = alpha_est
        self.beta = beta_est

    def marginalized_params(self, indices: list[int]) -> dict[str, Tensor]:
        return {"alpha": self.alpha[indices], "beta": self.beta[indices]}
