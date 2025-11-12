import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import (
    LeafModule,
    LogSpaceParameter,
    validate_all_or_none, init_parameter, parse_leaf_args,
)


class LogNormal(LeafModule):
    std = LogSpaceParameter("std")

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

        validate_all_or_none(mean=mean, std=std)

        mean = init_parameter(param=mean, event_shape=event_shape, init=torch.randn)
        std = init_parameter(param=std, event_shape=event_shape, init=torch.rand)

        self.mean = nn.Parameter(mean)
        self.log_std = nn.Parameter(
            torch.empty_like(std)
        )  # initialize empty, set with descriptor in next line
        self.std = std.clone().detach()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.LogNormal(self.mean, self.std)

    @property
    def _supported_value(self):
        return 1.0

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Estimate LogNormal mean and std and assign parameters.

        Args:
            data: Scope-filtered data of shape (batch_size, num_scope_features).
            weights: Normalized weights of shape (batch_size, 1, ...).
            bias_correction: Whether to apply Bessel's correction (n-1 vs n).
        """
        n_total = weights.sum()

        log_data = data.log()
        mean_est = (weights * log_data).sum(0) / n_total

        var_numerator = (weights * torch.pow(log_data - mean_est, 2)).sum(0)
        denom = n_total - 1 if bias_correction else n_total
        std_est = torch.sqrt(var_numerator / denom)

        # Broadcast to event_shape and assign directly
        self.mean.data = self._broadcast_to_event_shape(mean_est)
        self.std = self._broadcast_to_event_shape(std_est)

    def params(self) -> dict[str, Tensor]:
        return {"mean": self.mean, "std": self.std}
