import torch
from torch import Tensor, nn

from .distribution import Distribution
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import init_parameter, _handle_mle_edge_cases, parse_leaf_args


class ExponentialDistribution(Distribution):
    """Exponential distribution for modeling time-between-events.

    Parameterized by rate λ > 0 (stored in log-space for numerical stability).
    """

    def __init__(self, rate: Tensor = None, event_shape: tuple[int, ...] = None):
        """Initialize Exponential distribution.

        Args:
            rate: Rate parameter λ > 0.
            event_shape: The shape of the event. If None, it is inferred from rate shape.
        """
        if event_shape is None:
            event_shape = rate.shape
        super().__init__(event_shape=event_shape)

        rate = init_parameter(param=rate, event_shape=event_shape, init=torch.rand)

        # Validate rate at initialization
        if not (rate > 0).all():
            raise ValueError("Rate must be strictly positive")
        if not torch.isfinite(rate).all():
            raise ValueError("Rate must be finite")

        self.log_rate = nn.Parameter(torch.log(rate))

    @property
    def rate(self) -> Tensor:
        """Rate parameter in natural space (read via exp of log_rate)."""
        return torch.exp(self.log_rate)

    @rate.setter
    def rate(self, value: Tensor) -> None:
        """Set rate parameter (stores as log_rate, no validation after init)."""
        self.log_rate.data = torch.log(
            torch.as_tensor(value, dtype=self.log_rate.dtype, device=self.log_rate.device)
        )

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 0.0

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying Exponential distribution."""
        return torch.distributions.Exponential(self.rate)

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"rate": self.rate}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for rate parameter λ.

        For Exponential distribution, the MLE is λ = n / sum(x_i).

        Args:
            data: Scope-filtered data.
            weights: Normalized sample weights.
            bias_correction: Whether to apply bias correction (n-1 instead of n).
        """
        n_total = weights.sum()

        if bias_correction:
            n_total = n_total - 1

        rate_est = n_total / (weights * data).sum(0)

        # Handle edge cases (NaN, zero, or near-zero rate) before broadcasting
        rate_est = _handle_mle_edge_cases(rate_est, lb=0.0)

        # Broadcast to event_shape and assign - LogSpaceParameter ensures positivity
        self.rate = self._broadcast_to_event_shape(rate_est)


class Exponential(LeafModule):
    """Exponential distribution leaf for modeling time-between-events.

    Parameterized by rate λ > 0 (stored in log-space).

    Attributes:
        rate: Rate parameter λ (LogSpaceParameter).
        distribution: Underlying torch.distributions.Exponential.
    """

    def __init__(
        self, scope: Scope, out_channels: int = None, num_repetitions: int = None, rate: Tensor = None
    ):
        """Initialize Exponential distribution leaf.

        Args:
            scope: Variable scope.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions.
            rate: Rate parameter λ > 0.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[rate], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape
        self._distribution = ExponentialDistribution(rate=rate, event_shape=event_shape)
