import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule, BoundedParameter, init_parameter, parse_leaf_args


class Geometric(LeafModule):
    """Geometric distribution leaf for modeling trials until first success.

    Attributes:
        p (BoundedParameter): Success probability in (0, 1].
        distribution: Underlying torch.distributions.Geometric object.
    """

    p = BoundedParameter("p", lb=0.0, ub=1.0)

    def __init__(self, scope: Scope, out_channels: int = None, num_repetitions: int = None, p: Tensor = None):
        """Initialize Geometric distribution leaf.

        Args:
            scope: Variable scope for this distribution.
            out_channels: Number of output channels.
            num_repetitions: Number of repetitions.
            p: Success probability in (0, 1].
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[p], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        p = init_parameter(param=p, event_shape=event_shape, init=torch.rand)

        self.log_p = nn.Parameter(torch.empty_like(p))  # initialize empty, set with setter in next line
        self.p = p.clone().detach()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Geometric(self.p)

    @property
    def _supported_value(self):
        return 1

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for success probability p.

        Args:
            data: Scope-filtered data.
            weights: Normalized sample weights.
            bias_correction: Whether to apply bias correction.
        """
        n_total = weights.sum()
        n_success = (weights * data).sum(0)

        p_est = n_total / (n_success + n_total)
        if bias_correction:
            p_est = p_est - (p_est * (1 - p_est) / n_total)

        # Broadcast to event_shape and assign - BoundedParameter ensures [0, 1]
        self.p = self._broadcast_to_event_shape(p_est)

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"p": self.p}
