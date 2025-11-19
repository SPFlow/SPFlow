import torch
from torch import Tensor, nn

from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import init_parameter, _handle_mle_edge_cases


class Exponential(LeafModule):
    """Exponential distribution leaf for modeling time-between-events.

    Parameterized by rate λ > 0 (stored in log-space for numerical stability).

    Attributes:
        rate: Rate parameter λ (accessed via property, stored as log_rate).
        distribution: Underlying torch.distributions.Exponential.
    """

    def __init__(
            self,
            scope,
            out_channels: int = None,
            num_repetitions: int = None,
            parameter_network: nn.Module = None,
            validate_args: bool | None = True,
            rate: Tensor = None,
    ):
        """Initialize Exponential distribution leaf.

        Args:
            scope: Variable scope (Scope, int, or list[int]).
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions (for 3D event shapes).
            parameter_network: Optional neural network for parameter generation.
            validate_args: Whether to enable torch.distributions argument validation.
            rate: Rate parameter λ > 0.
        """
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[rate],
            parameter_network=parameter_network,
            validate_args=validate_args,
        )

        rate = init_parameter(param=rate, event_shape=self._event_shape, init=torch.rand)

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
    def _torch_distribution_class(self) -> type[torch.distributions.Exponential]:
        return torch.distributions.Exponential

    
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
