import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import (
    LeafModule,
    BoundedParameter, init_parameter, parse_leaf_args,
)


class Binomial(LeafModule):
    """
    Binomial distribution.
    """

    p = BoundedParameter("p", lb=0.0, ub=1.0)

    def __init__(
        self, scope: Scope, n: Tensor, out_channels: int = None, num_repetitions: int = None, p: Tensor = None
    ):
        r"""
        Initialize a Binomial distribution leaf module.

        Args:
            scope: Scope object specifying the scope of the distribution.
            n: Tensor containing the number (:math:`n`) of total trials.
            out_channels: Number of output channels. If None, it will be inferred from `p`.
            num_repetitions: Number of repetitions for the distribution.
            p: Tensor containing the success probability (:math:`p`) of each trial in :math:`[0,1]`.
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
        self.log_p = nn.Parameter(torch.empty_like(p))  # initialize empty, set with setter in next line
        self.p = p.clone().detach()

    @property
    def n(self) -> Tensor:
        """Returns the number of trials."""
        return self._n

    @n.setter
    def n(self, n: Tensor):
        r"""Sets the number of trials.

        Args:
            n:
                Floating point representing the number of trials.

        Raises:
            ValueError: Invalid arguments.
        """

        if torch.any(n < 0.0) or not torch.isfinite(n).all():
            raise ValueError(
                f"Value of 'n' for 'Binomial' distribution must be non-negative and finite, but was: {n}"
            )

        self._n = n

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Binomial(total_count=self.n, probs=self.p)

    @property
    def _supported_value(self):
        return 0.0

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Estimate Binomial success probability and assign.

        Args:
            data: Scope-filtered data of shape (batch_size, num_scope_features).
            weights: Normalized weights of shape (batch_size, 1, ...).
            bias_correction: Not used for Binomial (included for template consistency).
        """
        normalized_weights = weights / weights.sum()
        n_total = normalized_weights.sum() * self.n
        n_success = (normalized_weights * data).sum(0)
        success_est = self._broadcast_to_event_shape(n_success)
        p_est = success_est / n_total

        # p_est is already in the correct shape (broadcast=False in old code)
        # Clamp to [0, 1] before assigning to handle floating point precision issues
        p_est = self._handle_mle_edge_cases(p_est, lb=0.0, ub=1.0)
        # Assign directly - BoundedParameter ensures [0, 1]
        self.p = p_est

    def params(self) -> dict[str, Tensor]:
        return {"n": self.n, "p": self.p}
