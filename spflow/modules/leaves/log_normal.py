import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.leaves.base import (
    LeafModule,
    LogSpaceParameter,
    validate_all_or_none,
    init_parameter,
    parse_leaf_args,
)


class LogNormal(LeafModule):
    """Log-Normal distribution leaf for modeling positive-valued data.

    Note: Parameters μ and σ apply to ln(x), not x itself.

    Attributes:
        mean (Parameter): Mean μ of log-space distribution.
        std (LogSpaceParameter): Standard deviation σ > 0 of log-space distribution.
        distribution: Underlying torch.distributions.LogNormal object.
    """

    std = LogSpaceParameter("std")

    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        mean: Tensor = None,
        std: Tensor = None,
    ):
        """Initialize Log-Normal distribution leaf.

        Args:
            scope: Variable scope for this distribution.
            out_channels: Number of output channels.
            num_repetitions: Number of repetitions.
            mean: Mean μ of log-space distribution.
            std: Standard deviation σ > 0 of log-space distribution.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[mean, std], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        validate_all_or_none(mean=mean, std=std)

        mean = init_parameter(param=mean, event_shape=event_shape, init=torch.randn)
        std = init_parameter(param=std, event_shape=event_shape, init=torch.rand)

        self.mean = nn.Parameter(mean)
        self.log_std = nn.Parameter(
            torch.empty_like(std)
        )  # initialize empty, set with descriptor in next line
        self.std = std.clone().detach()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.LogNormal(self.mean, self.std)

    @property
    def _supported_value(self):
        return 1.0

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for mean μ and std σ by fitting ln(data).

        Args:
            data: Scope-filtered data (must be positive).
            weights: Normalized sample weights.
            bias_correction: Whether to apply bias correction.
        """
        n_total = weights.sum()

        log_data = data.log()
        mean_est = (weights * log_data).sum(0) / n_total

        var_numerator = (weights * torch.pow(log_data - mean_est, 2)).sum(0)
        denom = n_total - 1 if bias_correction else n_total
        std_est = torch.sqrt(var_numerator / denom)

        # Broadcast to event_shape and assign directly
        self.mean.data = self._broadcast_to_event_shape(mean_est)
        self.std = self._broadcast_to_event_shape(std_est)

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"mean": self.mean, "std": self.std}
