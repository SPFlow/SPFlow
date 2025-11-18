import torch
from torch import Tensor, nn

from spflow.distributions.base import Distribution
from spflow.utils.leaves import BoundedParameter, init_parameter, _handle_mle_edge_cases


class Binomial(Distribution):
    """Binomial distribution for modeling successes in fixed trials.

    Implements univariate Binomial distributions for probabilistic circuits.
    The Binomial distribution models the number of successes in a fixed number
    of independent Bernoulli trials, with probability mass function:
        P(X = k | n, p) = C(n, k) * p^k * (1-p)^(n-k)

    where n is the number of trials (fixed), p is the success probability (learnable),
    and k is the number of successes (0 ≤ k ≤ n).

    Attributes:
        p: Success probability parameter(s) in [0, 1] (BoundedParameter).
        n: Number of trials parameter(s), non-negative integers (fixed buffer).
    """

    p = BoundedParameter("p", lb=0.0, ub=1.0)

    def __init__(self, n: Tensor, p: Tensor = None, event_shape: tuple[int, ...] = None):
        """Initialize Binomial distribution.

        Args:
            n: Tensor containing the number (n) of total trials (fixed, non-negative).
            p: Tensor containing the success probability (p) of each trial in [0, 1].
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
        """Returns the number of trials."""
        return self._n

    @n.setter
    def n(self, n: Tensor):
        """Sets the number of trials.

        Args:
            n: Floating point representing the number of trials.

        Raises:
            ValueError: If n is not non-negative and finite.
        """
        if torch.any(n < 0.0) or not torch.isfinite(n).all():
            raise ValueError(
                f"Value of 'n' for 'Binomial' distribution must be non-negative and finite, but was: {n}"
            )
        self._n = n

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 0.0

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying Binomial distribution."""
        return torch.distributions.Binomial(total_count=self.n, probs=self.p)

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"n": self.n, "p": self.p}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for success probability p.

        Estimates the success probability parameter p using weighted maximum
        likelihood estimation. The parameter n is fixed and not learned.

        Args:
            data: Scope-filtered data.
            weights: Normalized weights.
            bias_correction: Not used for Binomial (included for interface consistency).
        """
        normalized_weights = weights / weights.sum()
        n_total = normalized_weights.sum() * self.n
        n_success = (normalized_weights * data).sum(0)
        success_est = self._broadcast_to_event_shape(n_success)
        p_est = success_est / n_total

        # Handle edge cases before assigning
        p_est = _handle_mle_edge_cases(p_est, lb=0.0, ub=1.0)

        # Assign directly - BoundedParameter ensures [0, 1]
        self.p = p_est
