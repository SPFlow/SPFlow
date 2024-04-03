#!/usr/bin/env python3

from typing import List

import torch
from torch import Tensor, nn

from spflow.distributions.distribution import Distribution
from spflow.meta.data import FeatureContext, FeatureTypes
from spflow.meta.data.meta_type import MetaType


class Categorical(Distribution):
    def __init__(self, p: Tensor, event_shape: tuple[int, ...] = None):
        r"""Initializes ``Categorical`` leaf node.

        Args:
            scope: Scope object specifying the scope of the distribution.
            p: Tensor containing the event probabilities of the distribution. Has shape (event_shape, k) where k is the number of categories.
            n_out: Number of nodes per scope. Only relevant if mean and std is None.
        """
        if event_shape is None:
            event_shape = p.shape[:2]
        super().__init__(event_shape=event_shape)

        self.log_p = nn.Parameter(torch.empty_like(p))  # initialize empty, set with setter in next line
        self.p = p.clone().detach()

    @property
    def p(self) -> Tensor:
        """Returns the probabilities."""
        return self.log_p.exp()

    @p.setter
    def p(self, p):
        """Set the probabilities."""
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(p).all():
            raise ValueError(f"Values for 'std' must be finite, but was: {p}")

        if torch.all(p <= 0.0):
            raise ValueError(f"Value for 'std' must be greater than 0.0, but was: {p}")

        # make sure that p adds up to 1
        if len(self.event_shape) == 1:
            p = p / p.sum(dim=-1)

        else:
            p = p / p.sum(dim=-1).unsqueeze(2)

        self.log_p.data = p.log()


    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(self.p)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        feature_ctx = signatures[0]
        domains = feature_ctx.get_domains()

        # leaf is a single non-conditional univariate node
        if len(domains) != 1 or len(feature_ctx.scope.query) != len(domains) or len(feature_ctx.scope.evidence) != 0:
            return False

        # leaf is a discrete Categorical distribution
        if not (
                domains[0] == FeatureTypes.Discrete
                or domains[0] == FeatureTypes.Categorical
                or isinstance(domains[0], FeatureTypes.Categorical)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "Categorical":
        if not cls.accepts(signatures):
            raise ValueError(f"'Categorical' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Discrete:
            k = 2
            p = [0.5, 0.5]
        elif domain == FeatureTypes.Categorical:
            # instantiate object
            k = domain().k
            p = domain().p
        elif isinstance(domain, FeatureTypes.Categorical):
            k = domain.k
            p = domain.p
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Categorical' that was not caught during acception checking."
            )

        return Categorical(p=p)

    def maximum_likelihood_estimation(self, data: Tensor, weights: Tensor = None, bias_correction=True):
        if weights is None:
            _shape = (data.shape[0], *([1] * (data.dim() - 1)))  # (batch, 1, 1, ...) for broadcasting
            weights = torch.ones(_shape, device=data.device)

        # total (weighted) number of instances
        n_total = weights.sum()

        # count (weighted) number of total successes
        n_success = (weights * data).sum(dim=0)

        p_est = []
        for column in range(data.shape[1]):
            p_k_est = []
            for cat in range(len(torch.unique(data))):
                cat_indices = data[:, column] == cat
                cat_data = cat_indices.float()
                cat_est = torch.sum(weights[0] * cat_data)
                cat_est /= n_total
                p_k_est.append(cat_est)
            p_est.append(p_k_est)

        p_est = torch.tensor(p_est)

        # edge case (if all values are the same, not enough samples or very close to each other)
        if torch.any(zero_mask := torch.isclose(p_est, torch.tensor(0.0))):
            p_est[zero_mask] = torch.tensor(1e-8)
        if torch.any(nan_mask := torch.isnan(p_est)):
            p_est[nan_mask] = torch.tensor(1e-8)

        if len(self.event_shape) == 2:
            # Repeat mean and std
            p_est = p_est.unsqueeze(1).repeat(1, self.event_shape[1], 1)

        # set parameters of leaf node and make sure they add up to 1
        self.p = p_est

    def marginalized_params(self, indices: list[int]) -> dict[str, Tensor]:
        return {"p": self.p[indices]}
