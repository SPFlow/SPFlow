import torch
from torch import Tensor, nn

from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import validate_all_or_none, init_parameter, _handle_mle_edge_cases


class LogNormal(LeafModule):
    """Log-Normal distribution leaf for modeling positive-valued data.

    Note: Parameters μ and σ apply to ln(x), not x itself.
    Standard deviation σ is stored in log-space for numerical stability.

    Attributes:
        mean: Mean μ of log-space distribution.
        std: Standard deviation σ > 0 of log-space distribution (via exp of log_std).
        distribution: Underlying torch.distributions.LogNormal.
    """

    def __init__(
        self,
        scope,
        out_channels: int = None,
        num_repetitions: int = 1,
        parameter_network: nn.Module = None,
        validate_args: bool | None = True,
        loc: Tensor = None,
        scale: Tensor = None,
    ):
        """Initialize Log-Normal distribution.

        Args:
            scope: Variable scope (Scope, int, or list[int]).
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions (for 3D event shapes).
            parameter_network: Optional neural network for parameter generation.
            validate_args: Whether to enable torch.distributions argument validation.
            loc: Mean μ of log-space distribution.
            scale: Standard deviation σ > 0 of log-space distribution.
        """
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[loc, scale],
            parameter_network=parameter_network,
            validate_args=validate_args,
        )

        validate_all_or_none(loc=loc, scale=scale)

        loc = init_parameter(param=loc, event_shape=self._event_shape, init=torch.randn)
        scale = init_parameter(param=scale, event_shape=self._event_shape, init=torch.rand)

        self.loc = nn.Parameter(loc)
        self.log_scale = nn.Parameter(torch.log(scale))

    @property
    def scale(self) -> Tensor:
        """Standard deviation in natural space (read via exp of log_scale)."""
        return torch.exp(self.log_scale)

    @scale.setter
    def scale(self, value: Tensor) -> None:
        """Set standard deviation (stores as log_scale, no validation after init)."""
        self.log_scale.data = torch.log(
            torch.as_tensor(value, dtype=self.log_scale.dtype, device=self.log_scale.device)
        )

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 1.0

    @property
    def _torch_distribution_class(self) -> type[torch.distributions.LogNormal]:
        return torch.distributions.LogNormal

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"loc": self.loc, "scale": self.scale}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for mean μ and std σ by fitting ln(data).

        For Log-Normal distribution, we compute statistics on log-transformed data.

        Args:
            data: Scope-filtered data (must be positive).
            weights: Normalized sample weights.
            bias_correction: Whether to apply bias correction to variance estimate.
        """
        n_total = weights.sum()

        log_data = data.log()
        loc_est = (weights * log_data).sum(0) / n_total

        var_numerator = (weights * torch.pow(log_data - loc_est, 2)).sum(0)
        denom = n_total - 1 if bias_correction else n_total
        std_est = torch.sqrt(var_numerator / denom)

        # Handle edge cases (NaN, zero, or near-zero std) before broadcasting
        std_est = _handle_mle_edge_cases(std_est, lb=0.0)

        # Broadcast to event_shape and assign directly
        self.loc.data = self._broadcast_to_event_shape(loc_est)
        self.scale = self._broadcast_to_event_shape(std_est)
