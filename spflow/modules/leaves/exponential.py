import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import LogSpaceParameter, init_parameter, parse_leaf_args


class Exponential(LeafModule):
    """Exponential distribution leaf for modeling time-between-events.

    Attributes:
        rate (LogSpaceParameter): Rate parameter λ > 0.
        distribution: Underlying torch.distributions.Exponential object.
    """

    rate = LogSpaceParameter("rate")

    def __init__(
        self, scope: Scope, out_channels: int = None, num_repetitions: int = None, rate: Tensor = None
    ):
        """Initialize Exponential distribution leaf.

        Args:
            scope: Variable scope for this distribution.
            out_channels: Number of output channels.
            num_repetitions: Number of repetitions.
            rate: Rate parameter λ.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[rate], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        rate = init_parameter(param=rate, event_shape=event_shape, init=torch.rand)

        self.log_rate = nn.Parameter(
            torch.empty_like(rate)
        )  # initialize empty, set with descriptor in next line
        self.rate = rate.clone().detach()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Exponential(self.rate)

    @property
    def _supported_value(self):
        return 0.0

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for rate parameter λ.

        Args:
            data: Scope-filtered data.
            weights: Normalized sample weights.
            bias_correction: Whether to apply bias correction.
        """
        n_total = weights.sum()

        if bias_correction:
            n_total = n_total - 1

        rate_est = n_total / (weights * data).sum(0)
        # Broadcast to event_shape and assign - LogSpaceParameter ensures positivity
        self.rate = self._broadcast_to_event_shape(rate_est)

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"rate": self.rate}
