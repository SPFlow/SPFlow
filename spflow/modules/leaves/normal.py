import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.leaves.base import (
    LeafModule,
)
from utils.leaves import LogSpaceParameter, validate_all_or_none, init_parameter, parse_leaf_args


class Normal(LeafModule):
    """Normal (Gaussian) distribution leaf module.

    Parameterized by mean μ and standard deviation σ (stored in log-space).

    Attributes:
        mean: Mean parameter.
        std: Standard deviation (LogSpaceParameter).
        distribution: Underlying torch.distributions.Normal.
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
        """Initialize Normal distribution leaf.

        Args:
            scope: Variable scope.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions.
            mean: Mean parameter tensor (random init if None).
            std: Standard deviation tensor (must be positive, random init if None).
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, num_repetitions=num_repetitions, params=[mean, std]
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

    def mode(self) -> Tensor:
        """Return mode (equals mean for Normal distribution)."""
        return self.mean

    @property
    def _supported_value(self):
        """Return supported value for edge case handling."""
        return 0.0

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Return underlying torch.distributions.Normal."""
        return torch.distributions.Normal(self.mean, self.std)

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute weighted mean and standard deviation.

        Args:
            data: Input data tensor.
            weights: Weight tensor for each data point.
            bias_correction: Whether to apply bias correction to variance estimate.
        """
        n_total = weights.sum()
        mean_est = (weights * data).sum(0) / n_total

        centered = data - mean_est
        var_numerator = (weights * centered.pow(2)).sum(0)
        denom = n_total - 1 if bias_correction else n_total
        std_est = torch.sqrt(var_numerator / denom)

        # Handle edge cases (NaN, zero, or near-zero std) before broadcasting
        std_est = self._handle_mle_edge_cases(std_est, lb=0.0)

        # Broadcast to event_shape and assign directly
        self.mean.data = self._broadcast_to_event_shape(mean_est)
        self.std = self._broadcast_to_event_shape(std_est)

    def params(self) -> dict[str, Tensor]:
        """Return distribution parameters."""
        return {"mean": self.mean, "std": self.std}
