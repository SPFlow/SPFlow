import torch
from torch import Tensor, nn
from typing import Optional, Callable

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args, init_parameter
from spflow.utils.cache import Cache
from spflow.exceptions import InvalidParameterCombinationError


class Normal(LeafModule):
    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        mean: Tensor = None,
        std: Tensor = None,
    ):
        """
        Initialize a Normal distribution leaf module.

        Args:
            scope (Scope): The scope of the distribution.
            out_channels (int, optional): The number of output channels. If None, it is determined by the parameter tensors.
            num_repetitions (int, optional): The number of repetitions for the leaf module.
            mean (Tensor, optional): The mean parameter tensor.
            std (Tensor, optional): The standard deviation parameter tensor.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, num_repetitions=num_repetitions, params=[mean, std]
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        if not (mean is None and std is None) ^ (mean is not None and std is not None):
            raise InvalidParameterCombinationError("Either mean and std must be specified or neither.")

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

    def mode(self) -> Tensor:
        """Returns the mode of the distribution."""
        return self.mean

    @property
    def _supported_value(self):
        """Returns the supported values of the distribution."""
        return 0.0

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying torch distribution object."""
        return torch.distributions.Normal(self.mean, self.std)

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
        """Maximum likelihood estimation for Normal distribution parameters.

        Args:
            data: The input data tensor.
            weights: Optional weights tensor. If None, uniform weights are created.
            bias_correction: If True, use unbiased estimator for standard deviation.
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
        mean_est = (weights * data).sum(0) / n_total
        std_est = (weights * (data - mean_est) ** 2).sum(0)

        if bias_correction:
            std_est = torch.sqrt((weights * torch.pow(data - mean_est, 2)).sum(0) / (n_total - 1))
        else:
            std_est = torch.sqrt((weights * torch.pow(data - mean_est, 2)).sum(0) / n_total)

        # Handle edge cases using helper method
        std_est = self._handle_mle_edge_cases(std_est)

        # Broadcast to event_shape using helper method
        mean_est = self._broadcast_to_event_shape(mean_est)
        std_est = self._broadcast_to_event_shape(std_est)

        # set parameters of leaf node
        self.mean.data = mean_est
        self.std = std_est

    def params(self):
        """Returns the parameters of the distribution."""
        return {"mean": self.mean, "std": self.std}
