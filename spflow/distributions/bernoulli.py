import torch
from torch import Tensor, nn

from spflow.distributions.base import Distribution
from spflow.utils.leaves import BoundedParameter, init_parameter, _handle_mle_edge_cases


class Bernoulli(Distribution):
    """Bernoulli distribution for binary outcomes.

    Parameterized by success probability p ∈ [0, 1].

    Attributes:
        p: Success probability (BoundedParameter).
    """

    p = BoundedParameter("p", lb=0.0, ub=1.0)

    def __init__(self, p: Tensor = None, event_shape: tuple[int, ...] = None):
        """Initialize Bernoulli distribution.

        Args:
            p: Success probability tensor in [0, 1].
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
        return 0.0

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying Bernoulli distribution."""
        return torch.distributions.Bernoulli(self.p)

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"p": self.p}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for success probability p.

        For Bernoulli distribution, the MLE is the weighted proportion of successes.

        Args:
            data: Scope-filtered data.
            weights: Normalized sample weights.
            bias_correction: Not used for Bernoulli.
        """
        n_total = weights.sum()
        n_success = (weights * data).sum(dim=0)
        p_est = n_success / n_total

        # Handle edge cases (NaN, zero, or near-zero p) before broadcasting
        p_est = _handle_mle_edge_cases(p_est, lb=0.0)

        # Broadcast to event_shape and assign directly
        # BoundedParameter descriptor handles clamping to [0, 1]
        self.p = self._broadcast_to_event_shape(p_est)
