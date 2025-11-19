import torch
from torch import Tensor, nn

from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import validate_all_or_none, init_parameter, _handle_mle_edge_cases


class Gamma(LeafModule):
    """Gamma distribution leaf for modeling positive-valued continuous data.

    Parameterized by shape α > 0 and rate β > 0 (both stored in log-space for numerical stability).

    Attributes:
        alpha: Shape parameter α (accessed via property, stored as log_alpha).
        beta: Rate parameter β (accessed via property, stored as log_beta).
        distribution: Underlying torch.distributions.Gamma.
    """

    def __init__(
            self,
            scope,
            out_channels: int = None,
            num_repetitions: int = None,
            parameter_network: nn.Module = None,
            validate_args: bool | None = True,
            concentration: Tensor = None,
            rate: Tensor = None,
    ):
        """Initialize Gamma distribution leaf.

        Args:
            scope: Variable scope (Scope, int, or list[int]).
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions (for 3D event shapes).
            parameter_network: Optional neural network for parameter generation.
            validate_args: Whether to enable torch.distributions argument validation.
            concentration: Shape parameter α > 0.
            rate: Rate parameter β > 0.
        """
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[concentration, rate],
            parameter_network=parameter_network,
            validate_args=validate_args,
        )

        validate_all_or_none(concentration=concentration, rate=rate)

        concentration = init_parameter(param=concentration, event_shape=self._event_shape, init=torch.rand)
        rate = init_parameter(param=rate, event_shape=self._event_shape, init=torch.rand)

        self.log_concentration = nn.Parameter(torch.log(concentration))
        self.log_rate = nn.Parameter(torch.log(rate))

    @property
    def concentration(self) -> Tensor:
        """Shape parameter in natural space (read via exp of log_alpha)."""
        return torch.exp(self.log_concentration)

    @concentration.setter
    def concentration(self, value: Tensor) -> None:
        """Set shape parameter (stores as log_alpha, no validation after init)."""
        self.log_concentration.data = torch.log(
            torch.as_tensor(value, dtype=self.log_concentration.dtype, device=self.log_concentration.device)
        )

    @property
    def rate(self) -> Tensor:
        """Rate parameter in natural space (read via exp of log_beta)."""
        return torch.exp(self.log_rate)

    @rate.setter
    def rate(self, value: Tensor) -> None:
        """Set rate parameter (stores as log_beta, no validation after init)."""
        self.log_rate.data = torch.log(
            torch.as_tensor(value, dtype=self.log_rate.dtype, device=self.log_rate.device)
        )

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 1.0

    @property
    def _torch_distribution_class(self) -> type[torch.distributions.Gamma]:
        return torch.distributions.Gamma

  
    def conditional_distribution(self, evidence: Tensor) -> torch.distributions.Gamma:
        # Pass evidence to parameter network to get parameters
        params = self.parameter_network(evidence)

        # Apply exponential to ensure positive parameters and construct torch Gamma distribution
        return torch.distributions.Gamma(
            concentration=torch.exp(params["concentration"]),
            rate=torch.exp(params["rate"]),
            validate_args=self._validate_args,
        )

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"concentration": self.concentration, "rate": self.rate}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for shape α and rate β parameters.

        Uses moment-matching equations to estimate parameters with optional bias correction.

        Args:
            data: Scope-filtered data.
            weights: Normalized sample weights.
            bias_correction: Whether to apply bias correction.
        """
        n_total = weights.sum()

        data_log = data.log()
        mean_xlnx = (weights * data_log * data).sum(dim=0) / n_total
        mean_x = (weights * data).sum(dim=0) / n_total
        mean_ln_x = (weights * data_log).sum(dim=0) / n_total

        theta_est = mean_xlnx - mean_x * mean_ln_x
        concentration_est = mean_x / theta_est
        rate_est = 1 / theta_est

        if bias_correction:
            concentration_est = concentration_est - 1 / n_total * (
                    3 * concentration_est
                    - 2 / 3 * (concentration_est / (1 + concentration_est))
                    - 4 / 5 * (concentration_est / (1 + concentration_est) ** 2)
            )
            rate_est = rate_est * ((n_total - 1) / n_total)

        # Handle edge cases before broadcasting
        concentration_est = _handle_mle_edge_cases(concentration_est, lb=0.0)
        rate_est = _handle_mle_edge_cases(rate_est, lb=0.0)

        # Broadcast to event_shape and assign - LogSpaceParameter ensures positivity
        self.concentration = self._broadcast_to_event_shape(concentration_est)
        self.rate = self._broadcast_to_event_shape(rate_est)
