import torch
from torch import Tensor, nn
from typing import Optional, Callable

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args, init_parameter
from spflow.utils.cache import Cache


class Bernoulli(LeafModule):
    r"""
    Create a Bernoulli leaf module.
    """

    def __init__(self, scope: Scope, out_channels: int = None, num_repetitions: int = None, p: Tensor = None):
        r"""
        Args:
            scope (Scope): The scope of the leaf module.
            out_channels (int, optional): The number of output channels. If None, it is inferred from the shape of the parameter tensor.
            num_repetitions (int, optional): The number of repetitions for the leaf module.
            p (Tensor): PyTorch tensor representing the success probabilities of the Bernoulli distributions.
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
    def p(self) -> Tensor:
        """Returns the p parameters."""
        return torch.exp(self.log_p)

    @p.setter
    def p(self, p):
        """Set the p parameters."""
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(p).all():
            raise ValueError(f"Values for 'p' must be finite, but was: {p}")

        if torch.any(p < 0.0):
            raise ValueError(f"Value for 'p' must not be smaller than 0.0, but was: {p}")

        if torch.any(p > 1.0):
            raise ValueError(f"Value for 'p' must not be greater than 1.0, but was: {p}")

        self.log_p.data = p.log()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying torch distribution object."""
        return torch.distributions.Bernoulli(self.p)

    @property
    def _supported_value(self):
        """Returns the supported values of the distribution."""
        return 0.0

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Optional[Tensor] = None,
        bias_correction: bool = True,
        nan_strategy: Optional[str | Callable] = None,
        check_support: bool = True,
        cache: Cache | None = None,
        preprocess_data: bool = True,
    ) -> None:
        """Maximum likelihood estimation for Bernoulli distribution parameters.

        Args:
            data: The input data tensor.
            weights: Optional weights tensor. If None, uniform weights are created.
            bias_correction: Unused for Bernoulli, kept for API consistency.
            nan_strategy: Optional string or callable specifying how to handle missing data.
            check_support: Boolean value indicating whether to check data support.
            cache: Optional cache dictionary.
            preprocess_data: Boolean indicating whether to select relevant data for scope.
        """
        # Always select data relevant to this scope (same as log_likelihood does)
        data = data[:, self.scope.query]
        # Prepare weights using helper method
        weights = self._prepare_mle_weights(data, weights)

        # total (weighted) number of instances
        n_total = weights.sum()

        # count (weighted) number of total successes
        n_success = (weights * torch.nan_to_num(data, nan=0.0)).sum(dim=0)

        # estimate (weighted) success probability
        p_est = n_success / n_total

        # Handle edge cases using helper method
        p_est = self._handle_mle_edge_cases(p_est)

        # Broadcast to event_shape using helper method
        p_est = self._broadcast_to_event_shape(p_est)

        # set parameters of leaf node
        self.p = p_est

    def params(self) -> dict[str, Tensor]:
        """Returns the parameters of the distribution."""
        return {"p": self.p}
