import torch
from torch import Tensor, nn

from spflow.distributions.base import Distribution
from spflow.utils.leaves import LogSpaceParameter, init_parameter, _handle_mle_edge_cases


class Poisson(Distribution):
    """Poisson distribution for modeling event counts.

    Parameterized by rate λ > 0 (stored in log-space).

    Attributes:
        rate: Rate parameter λ (LogSpaceParameter).
    """

    rate = LogSpaceParameter("rate")

    def __init__(self, rate: Tensor = None, event_shape: tuple[int, ...] = None):
        """Initialize Poisson distribution.

        Args:
            rate: Rate parameter λ > 0.
            event_shape: The shape of the event. If None, it is inferred from rate shape.
        """
        if event_shape is None:
            event_shape = rate.shape
        super().__init__(event_shape=event_shape)

        rate = init_parameter(param=rate, event_shape=event_shape, init=torch.ones)

        self.log_rate = nn.Parameter(torch.empty_like(rate))  # initialize empty, set with setter in next line
        self.rate = rate.clone().detach()

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 0

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying Poisson distribution."""
        return torch.distributions.Poisson(self.rate)

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"rate": self.rate}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for rate parameter λ.

        For Poisson distribution, the MLE is simply the weighted mean of the data.

        Args:
            data: Scope-filtered data.
            weights: Normalized sample weights.
            bias_correction: Not used for Poisson.
        """
        n_total = weights.sum()
        rate_est = (weights * data).sum(dim=0) / n_total

        # Handle edge cases (NaN, zero, or near-zero rate) before broadcasting
        rate_est = _handle_mle_edge_cases(rate_est, lb=0.0)

        # Broadcast to event_shape and assign - LogSpaceParameter ensures positivity
        self.rate = self._broadcast_to_event_shape(rate_est)
