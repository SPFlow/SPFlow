import torch
from torch import Tensor, nn
from typing import Optional, Callable

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import (
    LeafModule,
    BoundedParameter,
    MLEBatch,
    MLEEstimates,
    MLEParameterEstimate,
)
from spflow.utils.leaf import parse_leaf_args, init_parameter
from spflow.utils.cache import Cache


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
            raise ValueError(f"Values for 'n' must be finite and greater than 0, but was: {n}")

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

        if torch.any(n < 1.0) or not torch.isfinite(n).any():  # ToDo is 0 a valid value for n?
            raise ValueError(
                f"Value of 'n' for 'Binomial' distribution must to be greater than 0, but was: {n}"
            )

        self._n = n

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Binomial(total_count=self.n, probs=self.p)

    @property
    def _supported_value(self):
        return 0.0

    def _use_distribution_support(self) -> bool:
        """Skip torch's support check due to shape broadcast issues; use custom check instead."""
        return False

    def _custom_support_mask(self, data: Tensor) -> Tensor:
        """Binomial support: {0, 1, 2, ..., n}."""
        n = self.n
        original_data_shape = data.shape

        # Add batch dimension to n
        n_expanded = n.unsqueeze(0)  # (1, features, channels) or (1, features, channels, repetitions)

        # If data has fewer dimensions than n, expand it for the comparison
        # This handles the case where data is (batch, features) but n is (1, features, channels)
        data_expanded = data
        num_added_dims = 0
        while data_expanded.dim() < n_expanded.dim():
            data_expanded = data_expanded.unsqueeze(-1)  # Add trailing dimensions
            num_added_dims += 1

        integer_mask = torch.remainder(data_expanded, 1) == 0
        range_mask = (data_expanded >= 0) & (data_expanded <= n_expanded)
        mask = integer_mask & range_mask

        # Reduce the mask back to the original data shape
        # After broadcasting, the mask includes extra dimensions, so we take the first element along those
        for _ in range(num_added_dims):
            mask = mask[..., 0]  # Select first element along last dimension

        return mask

    def _mle_compute_statistics(self, batch: MLEBatch) -> MLEEstimates:
        """Estimate Binomial success probabilities (p) using shared template helpers."""
        data = batch.data
        weights = batch.weights

        normalized_weights = weights / weights.sum()
        n_total = normalized_weights.sum() * self.n
        n_success = (normalized_weights * data).sum(0)
        success_est = self._broadcast_to_event_shape(n_success)
        p_est = success_est / n_total

        return {"p": MLEParameterEstimate(p_est, lb=0.0, ub=1.0, broadcast=False)}

    def params(self) -> dict[str, Tensor]:
        return {"n": self.n, "p": self.p}
