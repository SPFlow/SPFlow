import torch
from torch import Tensor, nn
from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import (
    LeafModule,
    LogSpaceParameter,
    validate_all_or_none,
)
from spflow.utils.leaf import parse_leaf_args, init_parameter


class Normal(LeafModule):
    std = LogSpaceParameter("std")

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

        validate_all_or_none(mean=mean, std=std)

        mean = init_parameter(param=mean, event_shape=event_shape, init=torch.randn)
        std = init_parameter(param=std, event_shape=event_shape, init=torch.rand)

        self.mean = nn.Parameter(mean)
        self.log_std = nn.Parameter(
            torch.empty_like(std)
        )  # initialize empty, set with descriptor in next line
        self.std = std.clone().detach()

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

    def _mle_compute_statistics(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> None:
        """Compute Normal-specific sufficient statistics and assign parameters.

        Args:
            data: Scope-filtered data of shape (batch_size, num_scope_features).
            weights: Normalized weights of shape (batch_size, 1, ...).
            bias_correction: Whether to apply Bessel's correction (n-1 vs n).
        """
        n_total = weights.sum()
        mean_est = (weights * data).sum(0) / n_total

        centered = data - mean_est
        var_numerator = (weights * centered.pow(2)).sum(0)
        denom = n_total - 1 if bias_correction else n_total
        std_est = torch.sqrt(var_numerator / denom)

        # Handle edge cases (NaN, zero, or near-zero std) before broadcasting
        std_est = self._handle_mle_edge_cases(std_est, lb=0.0)

        # Broadcast to event_shape and assign directly
        self.mean.data = self._broadcast_to_event_shape(mean_est)
        self.std = self._broadcast_to_event_shape(std_est)

    def params(self) -> dict[str, Tensor]:
        """Returns the parameters of the distribution."""
        return {"mean": self.mean, "std": self.std}
