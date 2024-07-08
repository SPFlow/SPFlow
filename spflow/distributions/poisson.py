import torch
from torch import Tensor, nn

from spflow.distributions.distribution import Distribution
from spflow.meta.data import FeatureContext, FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.utils.leaf import init_parameter


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

        rate = init_parameter(param=rate, event_shape=event_shape, init=torch.ones)

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
            rate_est = rate_est.unsqueeze(1).repeat(1, self.out_channels)

        # set parameters of leaf node
        self.rate = rate_est

    def params(self):
        return {"rate": self.rate}
