import torch
from torch import Tensor, nn
from typing import Optional, Callable

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args, init_parameter
from spflow.utils.cache import Cache


class Poisson(LeafModule):
    def __init__(
        self, scope: Scope, out_channels: int = None, num_repetitions: int = None, rate: Tensor = None
    ):
        r"""
        Initialize a Poisson distribution leaf module.

        Args:
            scope: Scope object specifying the scope of the distribution.
            out_channels: The number of output channels. If None, it is determined by the parameter tensor.
            num_repetitions: The number of repetitions for the leaf module.
            rate: Tensor representing the rate parameters (:math:`\lambda`) of the Poisson distributions.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[rate], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        rate = init_parameter(param=rate, event_shape=event_shape, init=torch.ones)

        self.log_rate = nn.Parameter(torch.empty_like(rate))  # initialize empty, set with setter in next line
        self.rate = rate.clone().detach()

    @property
    def rate(self) -> Tensor:
        """Returns the rate."""
        return self.log_rate.exp()

    @rate.setter
    def rate(self, rate):
        """Set the rate."""
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(rate).all():
            raise ValueError(f"Values for 'rate' must be finite, but was: {rate}")

        if torch.any(rate <= 0.0):
            raise ValueError(f"Value for 'rate' must be greater than 0.0, but was: {rate}")

        self.log_rate.data = rate.log()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Poisson(self.rate)

    @property
    def _supported_value(self):
        return 0

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
        """Maximum likelihood estimation for Poisson distribution parameters.

        Args:
            data: The input data tensor.
            weights: Optional weights tensor. If None, uniform weights are created.
            bias_correction: If True, apply bias correction to the estimate (not used for Poisson).
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

        # estimate rate parameter from data
        rate_est = (weights * torch.nan_to_num(data, nan=0.0)).sum(dim=0) / n_total

        # Handle edge cases using helper method
        rate_est = self._handle_mle_edge_cases(rate_est)

        # Broadcast to event_shape using helper method
        rate_est = self._broadcast_to_event_shape(rate_est)

        # set parameters of leaf node
        self.rate = rate_est

    def params(self):
        return {"rate": self.rate}
