import torch
from torch import Tensor, nn
from typing import Optional, Callable

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule, BoundedParameter, MLEBatch, MLEParameterEstimate
from spflow.utils.leaf import parse_leaf_args, init_parameter
from spflow.utils.cache import Cache


class NegativeBinomial(LeafModule):
    p = BoundedParameter("p", lb=0.0, ub=1.0)

    def __init__(
        self, scope: Scope, n: Tensor, out_channels: int = None, num_repetitions: int = None, p: Tensor = None
    ):
        r"""
        Initialize a NegativeBinomial distribution leaf module.

        Args:
            scope: Scope object specifying the scope of the distribution.
            out_channels: The number of output channels. If None, it is determined by the parameter tensors.
            num_repetitions: The number of repetitions for the leaf module.
            n: Tensor representing the numbers of  successes (greater or equal to 0).
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
        self.log_p = nn.Parameter(torch.empty_like(p))  # initialize empty, set with descriptor in next line
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
        return torch.distributions.NegativeBinomial(total_count=self.n, probs=self.p)

    @property
    def _supported_value(self):
        return 0

    def _mle_compute_statistics(self, batch: MLEBatch) -> dict[str, MLEParameterEstimate]:
        """Estimate success probability for the Negative Binomial distribution."""
        data = batch.data
        weights = batch.weights

        n_total = weights.sum() * self.n
        if batch.bias_correction:
            n_total = n_total - 1

        n_success = (weights * data).sum(0)
        success_est = self._broadcast_to_event_shape(n_success)
        p_est = 1 - n_total / (success_est + n_total)

        return {"p": MLEParameterEstimate(p_est, lb=0.0, ub=1.0, broadcast=False)}

    def params(self) -> dict[str, Tensor]:
        return {"n": self.n, "p": self.p}
