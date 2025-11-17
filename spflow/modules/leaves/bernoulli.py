import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from utils.leaves import BoundedParameter, init_parameter, parse_leaf_args


class Bernoulli(LeafModule):
    """Bernoulli distribution leaf module.

    Binary random variable with success probability p in [0, 1].

    Attributes:
        p: Success probability (BoundedParameter).
        distribution: Underlying torch.distributions.Bernoulli.
    """

    p = BoundedParameter("p", lb=0.0, ub=1.0)

    def __init__(self, scope: Scope, out_channels: int = None, num_repetitions: int = None, p: Tensor = None):
        """Initialize Bernoulli distribution leaf.

        Args:
            scope: Variable scope.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions.
            p: Success probability tensor in [0, 1] (random init if None).
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
        """Return underlying torch.distributions.Bernoulli."""
        return torch.distributions.Bernoulli(self.p)

    @property
    def _supported_value(self):
        """Return supported value for edge case handling."""
        return 0.0

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute weighted success probability.

        Args:
            data: Input data tensor.
            weights: Weight tensor for each data point.
            bias_correction: Whether to apply bias correction.
        """
        n_total = weights.sum()
        n_success = (weights * data).sum(dim=0)
        p_est = n_success / n_total

        # Broadcast to event_shape and assign directly
        # BoundedParameter descriptor handles clamping to [0, 1]
        self.p = self._broadcast_to_event_shape(p_est)

    def params(self) -> dict[str, Tensor]:
        """Return distribution parameters."""
        return {"p": self.p}
