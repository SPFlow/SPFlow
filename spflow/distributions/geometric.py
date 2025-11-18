import torch
from torch import Tensor, nn

from spflow.distributions.base import Distribution
from spflow.utils.leaves import BoundedParameter, init_parameter, _handle_mle_edge_cases


class Geometric(Distribution):
    """Geometric distribution for modeling trials until first success.

    Parameterized by success probability p ∈ (0, 1].

    Attributes:
        p: Success probability (BoundedParameter).
    """

    p = BoundedParameter("p", lb=0.0, ub=1.0)

    def __init__(self, p: Tensor = None, event_shape: tuple[int, ...] = None):
        """Initialize Geometric distribution.

        Args:
            p: Success probability tensor in (0, 1].
            event_shape: The shape of the event. If None, it is inferred from p shape.
        """
        if event_shape is None:
            event_shape = p.shape
        super().__init__(event_shape=event_shape)

        p = init_parameter(param=p, event_shape=event_shape, init=torch.rand)

        self.log_p = nn.Parameter(torch.empty_like(p))  # initialize empty, set with setter in next line
        self.p = p.clone().detach()

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 1

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying Geometric distribution."""
        return torch.distributions.Geometric(self.p)

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"p": self.p}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for success probability p.

        For Geometric distribution, the MLE is p = n / (sum(x_i) + n).

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

        # Handle edge cases (NaN, zero, or near-zero p) before broadcasting
        p_est = _handle_mle_edge_cases(p_est, lb=0.0)

        # Broadcast to event_shape and assign - BoundedParameter ensures [0, 1]
        self.p = self._broadcast_to_event_shape(p_est)
