#!/usr/bin/env python3

import torch
from torch import Tensor, nn

from spflow.distributions.distribution import Distribution
from spflow.meta.data import FeatureContext, FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.modules.node.leaf.utils import init_parameter


class Normal(Distribution):
    def __init__(self, mean: Tensor = None, std: Tensor = None, event_shape: tuple[int, ...] = None):
        r"""Initializes ``Normal`` leaf node.

        Args:
            scope: Scope object specifying the scope of the distribution.
            mean: Tensor containing the mean (:math:`\mu`) of the distribution.
            std: Tensor containing the standard deviation (:math:`\sigma`) of the distribution.
            n_out: Number of nodes per scope. Only relevant if mean and std is None.
        """
        if event_shape is None:
            event_shape = mean.shape
        super().__init__(event_shape=event_shape)
        assert (mean is None and std is None) ^ (
            mean is not None and std is not None
        ), "Either mean and std must be specified or neither."

        mean = init_parameter(param=mean, event_shape=event_shape, init=torch.randn)
        std = init_parameter(param=std, event_shape=event_shape, init=torch.rand)

        self.mean = nn.Parameter(mean)
        self.log_std = nn.Parameter(torch.empty_like(std))  # initialize empty, set with setter in next line
        self.std = std.clone().detach()

    @property
    def std(self) -> Tensor:
        """Returns the standard deviation."""
        return self.log_std.exp()

    @std.setter
    def std(self, std):
        """Set the standard deviation."""
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(std).all():
            raise ValueError(f"Values for 'std' must be finite, but was: {std}")

        if torch.all(std <= 0.0):
            raise ValueError(f"Value for 'std' must be greater than 0.0, but was: {std}")

        self.log_std.data = std.log()

    def mode(self) -> Tensor:
        return self.mean

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Normal(self.mean, self.std)

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

        # leaf is a continuous Normal distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.Normal
            or isinstance(domains[0], FeatureTypes.Normal)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Normal":
        if not cls.accepts(signatures):
            raise ValueError(f"'Normal' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Continuous:
            mean, std = 0.0, 1.0  # TODO: adapt to event_shape
            # How do we get the event_shape here?
            # mean = torch.tensor(mean).view([1]* len(self.event_shape)).repeat(self.event_shape)
        elif domain == FeatureTypes.Normal:
            # instantiate object
            domain = domain()
            mean, std = domain.mean, domain.std
        elif isinstance(domain, FeatureTypes.Normal):
            mean, std = domain.mean, domain.std
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Normal' that was not caught during acception checking."
            )

        return Normal(mean=mean, std=std)

    def maximum_likelihood_estimation(self, data: Tensor, weights: Tensor = None, bias_correction=True):
        # TODO: make some assertions about the event_shape and the data
        if weights is None:
            _shape = (data.shape[0], *([1] * (data.dim() - 1)))  # (batch, 1, 1, ...) for broadcasting
            weights = torch.ones(_shape, device=data.device)

        # total (weighted) number of instances
        n_total = weights.sum()

        # calculate mean and standard deviation from data
        mean_est = (weights * data).sum(0) / n_total
        std_est = (weights * (data - mean_est) ** 2).sum(0)

        if bias_correction:
            std_est = torch.sqrt((weights * torch.pow(data - mean_est, 2)).sum(0) / (n_total - 1))
        else:
            std_est = torch.sqrt((weights * torch.pow(data - mean_est, 2)).sum(0) / n_total)

        # edge case (if all values are the same, not enough samples or very close to each other)
        if torch.any(zero_mask := torch.isclose(std_est, torch.tensor(0.0))):
            std_est[zero_mask] = torch.tensor(1e-8)
        if torch.any(nan_mask := torch.isnan(std_est)):
            std_est[nan_mask] = torch.tensor(1e-8)

        if len(self.event_shape) == 2:
            # Repeat mean and std
            mean_est = mean_est.unsqueeze(1).repeat(1, self.event_shape[1])
            std_est = std_est.unsqueeze(1).repeat(1, self.event_shape[1])

        # set parameters of leaf node
        self.mean.data = mean_est
        self.std = std_est

    def marginalized_params(self, indices: list[int]) -> dict[str, Tensor]:
        return {"mean": self.mean[indices], "std": self.std[indices]}
