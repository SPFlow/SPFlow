#!/usr/bin/env python3

import torch
from torch import Tensor, nn

from spflow.distributions.distribution import Distribution
from spflow.meta.data import FeatureContext, FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.modules.node.leaf.utils import init_parameter


class Poisson(Distribution):
    def __init__(self, rate: Tensor = None, event_shape: tuple[int, ...] = None):
        r"""Initializes ``Poisson`` leaf node.

        Args:
            scope: Scope object specifying the scope of the distribution.
            rate: Tensor representing the rate parameters (:math:`\lambda`) of the Poisson distributions.
            n_out: Number of nodes per scope. Only relevant if mean and rate is None.
        """
        if event_shape is None:
            event_shape = rate.shape
        super().__init__(event_shape=event_shape)

        rate = init_parameter(param=rate, event_shape=event_shape, init=lambda:torch.tensor(1.0))

        self.log_rate = nn.Parameter(torch.empty_like(rate))  # initialize empty, set with setter in next line
        self.rate = rate.clone().detach()

    @property
    def rate(self) -> Tensor:
        """Returns the rate."""
        return self.log_rate.exp()

    @rate.setter
    def rate(self, rate):
        """Set the rate."""
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(rate).all():
            raise ValueError(f"Values for 'rate' must be finite, but was: {rate}")

        if torch.all(rate <= 0.0):
            raise ValueError(f"Value for 'rate' must be greater than 0.0, but was: {rate}")

        self.log_rate.data = rate.log()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Poisson(self.rate)

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

        # leaf is a discrete Poisson distribution
        if not (
            domains[0] == FeatureTypes.Discrete
            or domains[0] == FeatureTypes.Poisson
            or isinstance(domains[0], FeatureTypes.Poisson)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Poisson":
        if not cls.accepts(signatures):
            raise ValueError(f"'Poisson' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Discrete:
            rate = 1.0
        elif domain == FeatureTypes.Poisson:
            # instantiate object
            rate = domain().rate
        elif isinstance(domain, FeatureTypes.Poisson):
            rate = domain.rate
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Poisson' that was not caught during acception checking."
            )

        return Poisson(rate=rate)

    def maximum_likelihood_estimation(self, data: Tensor, weights: Tensor = None, bias_correction=True):
        if weights is None:
            _shape = (data.shape[0], *([1] * (data.dim() - 1)))  # (batch, 1, 1, ...) for broadcasting
            weights = torch.ones(_shape, device=data.device)

        # total (weighted) number of instances
        n_total = weights.sum()

        # estimate rate parameter from data
        rate_est = (weights * torch.nan_to_num(data, nan=0.0)).sum(dim=0) / n_total

        # edge case (if all values are the same, not enough samples or very close to each other)
        if torch.any(zero_mask := torch.isclose(rate_est, torch.tensor(0.0))):
            rate_est[zero_mask] = torch.tensor(1e-8)
        if torch.any(nan_mask := torch.isnan(rate_est)):
            rate_est[nan_mask] = torch.tensor(1e-8)

        if len(self.event_shape) == 2:
            # Repeat rate
            rate_est = rate_est.unsqueeze(1).repeat(1, self.event_shape[1])

        # set parameters of leaf node
        self.rate = rate_est

    def marginalized_params(self, indices: list[int]) -> dict[str, Tensor]:
        return {"rate": self.rate[indices]}
