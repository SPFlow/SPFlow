import torch
from torch import Tensor, nn
from typing import Optional, Callable

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args, init_parameter
from spflow.exceptions import InvalidParameterCombinationError
from spflow.utils.cache import Cache


class LogNormal(LeafModule):
    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        mean: Tensor = None,
        std: Tensor = None,
    ):
        r"""
        Initialize a LogNormal distribution leaf module.

        Args:
            scope: Scope object specifying the scope of the distribution.
            out_channels: The number of output channels. If None, it is determined by the parameter tensors.
            num_repetitions: The number of repetitions for the leaf module.
            mean: Tensor containing the mean (:math:`\mu`) of the distribution.
            std: Tensor containing the standard deviation (:math:`\sigma`) of the distribution.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[mean, std], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        if not ((mean is None and std is None) ^ (mean is not None and std is not None)):
            raise InvalidParameterCombinationError(
                "Either mean and std must be specified or neither."
            )

        mean = init_parameter(param=mean, event_shape=event_shape, init=torch.randn)
        std = init_parameter(param=std, event_shape=event_shape, init=torch.rand)

        self.mean = nn.Parameter(mean)
        self.log_std = nn.Parameter(torch.empty_like(std))  # initialize empty, set with setter in next line
        self.std = std.clone().detach()

    @property
    def std(self) -> Tensor:
        """Returns the standard deviation."""
        return self.log_std.exp()

    @std.setter
    def std(self, std):
        """Set the standard deviation."""
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(std).all():
            raise ValueError(f"Values for 'std' must be finite, but was: {std}")

        if torch.any(std <= 0.0):
            raise ValueError(f"Value for 'std' must be greater than 0.0, but was: {std}")

        self.log_std.data = std.log()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.LogNormal(self.mean, self.std)

    @property
    def _supported_value(self):
        return 1.0

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
        """Maximum likelihood estimation for LogNormal distribution parameters.

        Args:
            data: The input data tensor.
            weights: Optional weights tensor. If None, uniform weights are created.
            bias_correction: If True, apply bias correction to the estimate.
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

        # calculate mean and standard deviation from data
        mean_est = (weights * data.log()).sum(0) / n_total

        if bias_correction:
            std_est = torch.sqrt((weights * torch.pow(data.log() - mean_est, 2)).sum(0) / (n_total - 1))
        else:
            std_est = torch.sqrt((weights * torch.pow(data.log() - mean_est, 2)).sum(0) / n_total)

        # Handle edge cases using helper method
        std_est = self._handle_mle_edge_cases(std_est)

        # Broadcast to event_shape using helper method
        mean_est = self._broadcast_to_event_shape(mean_est)
        std_est = self._broadcast_to_event_shape(std_est)

        # set parameters of leaf node
        self.mean.data = mean_est
        self.std = std_est

    def params(self):
        return {"mean": self.mean, "std": self.std}
