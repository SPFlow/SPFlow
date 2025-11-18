import torch
from torch import Tensor, nn

from spflow.distributions.base import Distribution
from spflow.utils.leaves import BoundedParameter, init_parameter, _handle_mle_edge_cases


class NegativeBinomial(Distribution):
    """Negative Binomial distribution for modeling failures before r-th success.

    Note: Parameter n (number of successes) is fixed and cannot be learned.

    Attributes:
        n: Fixed number of required successes (buffer).
        p: Success probability in [0, 1] (BoundedParameter).
    """

    p = BoundedParameter("p", lb=0.0, ub=1.0)

    def __init__(self, n: Tensor, p: Tensor = None, event_shape: tuple[int, ...] = None):
        """Initialize Negative Binomial distribution.

        Args:
            n: Fixed number of required successes (must be non-negative).
            p: Success probability tensor in [0, 1].
            event_shape: The shape of the event. If None, it is inferred from p shape.
        """
        if event_shape is None:
            event_shape = p.shape
        super().__init__(event_shape=event_shape)

        p = init_parameter(param=p, event_shape=event_shape, init=torch.rand)

        # Validate n parameter
        if not torch.isfinite(n).all() or n.lt(0.0).any():
            raise ValueError(f"Values for 'n' must be finite and non-negative, but was: {n}")

        # Register n as a fixed buffer
        n = torch.broadcast_to(n, event_shape).clone()
        self.register_buffer("_n", n)
        self.n = n

        self.log_p = nn.Parameter(torch.empty_like(p))  # initialize empty, set with setter in next line
        self.p = p.clone().detach()

    @property
    def n(self) -> Tensor:
        """Returns the fixed number of required successes."""
        return self._n

    @n.setter
    def n(self, n: Tensor):
        """Sets the number of required successes.

        Args:
            n: Non-negative number of required successes.

        Raises:
            ValueError: If n is not non-negative and finite.
        """
        if torch.any(n < 0.0) or not torch.isfinite(n).all():
            raise ValueError(
                f"Value of 'n' for 'NegativeBinomial' distribution must be non-negative and finite, but was: {n}"
            )
        self._n = n

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 0

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying Negative Binomial distribution."""
        return torch.distributions.NegativeBinomial(total_count=self.n, probs=self.p)

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"n": self.n, "p": self.p}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for success probability p (given fixed n).

        Args:
            data: Scope-filtered data (failure counts).
            weights: Normalized sample weights.
            bias_correction: Whether to apply bias correction.
        """
        n_total = weights.sum() * self.n
        if bias_correction:
            n_total = n_total - 1

        n_success = (weights * data).sum(0)
        success_est = self._broadcast_to_event_shape(n_success)
        p_est = 1 - n_total / (success_est + n_total)

        # Handle edge cases before assigning
        p_est = _handle_mle_edge_cases(p_est, lb=0.0, ub=1.0)

        # Assign directly - BoundedParameter ensures [0, 1]
        self.p = p_est
