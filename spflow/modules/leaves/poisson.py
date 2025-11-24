import torch
from torch import Tensor, nn

from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import init_parameter, _handle_mle_edge_cases


class Poisson(LeafModule):
    """Poisson distribution leaf for modeling event counts.

    Parameterized by rate λ > 0 (stored in log-space for numerical stability).

    Attributes:
        rate: Rate parameter λ (stored as log_rate internally).
        distribution: Underlying torch.distributions.Poisson.
    """

    def __init__(
        self,
        scope,
        out_channels: int = None,
        num_repetitions: int = 1,
        parameter_fn: nn.Module = None,
        validate_args: bool | None = True,
        rate: Tensor = None,
    ):
        """Initialize Poisson leaf.

        Args:
            scope: Variable scope (Scope, int, or list[int]).
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions (for 3D event shapes).
            parameter_fn: Optional neural network for parameter generation.
            validate_args: Whether to enable torch.distributions argument validation.
            rate: Rate parameter λ > 0.
        """
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[rate],
            parameter_fn=parameter_fn,
            validate_args=validate_args,
        )

        rate = init_parameter(param=rate, event_shape=self._event_shape, init=torch.ones)

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
        return 0

    @property
    def _torch_distribution_class(self) -> type[torch.distributions.Poisson]:
        return torch.distributions.Poisson

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"rate": self.rate}

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> dict[str, Tensor]:
        """Compute raw MLE estimates for Poisson distribution (without broadcasting).

        For Poisson distribution, the MLE is simply the weighted mean of the data.

        Args:
            data: Input data tensor.
            weights: Weight tensor for each data point.
            bias_correction: Not used for Poisson.

        Returns:
            Dictionary with 'rate' estimate (shape: out_features).
        """
        n_total = weights.sum()
        rate_est = (weights * data).sum(dim=0) / n_total

        # Handle edge cases (NaN, zero, or near-zero rate) before broadcasting
        rate_est = _handle_mle_edge_cases(rate_est, lb=0.0)

        return {"rate": rate_est}

    def _set_mle_parameters(self, params_dict: dict[str, Tensor]) -> None:
        """Set MLE-estimated parameters for Poisson distribution.

        Explicitly handles the parameter type:
        - rate: Property with setter, calls property setter which updates log_rate

        Args:
            params_dict: Dictionary with 'rate' parameter value.
        """
        self.rate = params_dict["rate"]  # Uses property setter

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for rate parameter λ.

        For Poisson distribution, the MLE is simply the weighted mean of the data.

        Args:
            data: Scope-filtered data.
            weights: Normalized sample weights.
            bias_correction: Not used for Poisson.
        """
        estimates = self._compute_parameter_estimates(data, weights, bias_correction)

        # Broadcast to event_shape and assign - property setter ensures positivity
        self.rate = self._broadcast_to_event_shape(estimates["rate"])
