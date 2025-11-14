import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule, BoundedParameter, init_parameter, parse_leaf_args


class NegativeBinomial(LeafModule):
    """Negative Binomial distribution leaf for modeling failures before r-th success.

    Note: Parameter n (number of successes) is fixed and cannot be learned.

    Attributes:
        n (Tensor): Fixed number of required successes (buffer).
        p (BoundedParameter): Success probability in [0, 1].
        distribution: Underlying torch.distributions.NegativeBinomial object.
    """

    p = BoundedParameter("p", lb=0.0, ub=1.0)

    def __init__(
        self, scope: Scope, n: Tensor, out_channels: int = None, num_repetitions: int = None, p: Tensor = None
    ):
        """Initialize Negative Binomial distribution leaf.

        Args:
            scope: Variable scope for this distribution.
            n: Fixed number of required successes (must be non-negative).
            out_channels: Number of output channels.
            num_repetitions: Number of repetitions.
            p: Success probability in [0, 1].
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[p], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        p = init_parameter(param=p, event_shape=event_shape, init=torch.rand)

        if not torch.isfinite(n).all() or n.lt(0.0).any():
            raise ValueError(f"Values for 'n' must be finite and non-negative, but was: {n}")

        n = torch.broadcast_to(n, event_shape).clone()
        self.register_buffer("_n", n)
        self.n = n
        self.log_p = nn.Parameter(torch.empty_like(p))  # initialize empty, set with descriptor in next line
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
        """
        if torch.any(n < 0.0) or not torch.isfinite(n).all():
            raise ValueError(
                f"Value of 'n' for 'NegativeBinomial' distribution must be non-negative and finite, but was: {n}"
            )

        self._n = n

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.NegativeBinomial(total_count=self.n, probs=self.p)

    @property
    def _supported_value(self):
        return 0

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
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

        # Clamp to [0, 1] before assigning to handle floating point precision issues
        p_est = self._handle_mle_edge_cases(p_est, lb=0.0, ub=1.0)
        # Assign directly - BoundedParameter ensures [0, 1]
        self.p = p_est

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"n": self.n, "p": self.p}
