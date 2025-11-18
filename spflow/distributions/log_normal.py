import torch
from torch import Tensor, nn

from spflow.distributions.base import Distribution
from spflow.utils.leaves import LogSpaceParameter, validate_all_or_none, init_parameter, _handle_mle_edge_cases


class LogNormal(Distribution):
    """Log-Normal distribution for positive-valued data.

    Note: Parameters μ and σ apply to ln(x), not x itself.

    Attributes:
        mean: Mean μ of log-space distribution.
        std: Standard deviation σ > 0 of log-space distribution (LogSpaceParameter).
    """

    std = LogSpaceParameter("std")

    def __init__(self, mean: Tensor = None, std: Tensor = None, event_shape: tuple[int, ...] = None):
        """Initialize Log-Normal distribution.

        Args:
            mean: Mean μ of log-space distribution.
            std: Standard deviation σ > 0 of log-space distribution.
            event_shape: The shape of the event. If None, it is inferred from parameter shapes.
        """
        if event_shape is None:
            event_shape = mean.shape
        super().__init__(event_shape=event_shape)

        validate_all_or_none(mean=mean, std=std)

        mean = init_parameter(param=mean, event_shape=event_shape, init=torch.randn)
        std = init_parameter(param=std, event_shape=event_shape, init=torch.rand)

        self.mean = nn.Parameter(mean)
        self.log_std = nn.Parameter(torch.empty_like(std))  # initialize empty, set with setter in next line
        self.std = std.clone().detach()

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 1.0

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying LogNormal distribution."""
        return torch.distributions.LogNormal(self.mean, self.std)

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"mean": self.mean, "std": self.std}

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
        mean_est = (weights * log_data).sum(0) / n_total

        var_numerator = (weights * torch.pow(log_data - mean_est, 2)).sum(0)
        denom = n_total - 1 if bias_correction else n_total
        std_est = torch.sqrt(var_numerator / denom)

        # Handle edge cases (NaN, zero, or near-zero std) before broadcasting
        std_est = _handle_mle_edge_cases(std_est, lb=0.0)

        # Broadcast to event_shape and assign directly
        self.mean.data = self._broadcast_to_event_shape(mean_est)
        self.std = self._broadcast_to_event_shape(std_est)
