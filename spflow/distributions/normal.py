import torch
from torch import Tensor, nn

from spflow.distributions.base import Distribution
from spflow.utils.leaves import validate_all_or_none, LogSpaceParameter, init_parameter, \
    _handle_mle_edge_cases


class Normal(Distribution):

    std = LogSpaceParameter("std")

    def __init__(self, mean: Tensor = None, std: Tensor = None, event_shape: tuple[int, ...] = None):
        r"""

        Args:
            mean: Tensor containing the mean (:math:`\mu`) of the distribution.
            std: Tensor containing the standard deviation (:math:`\sigma`) of the distribution.
            event_shape: The shape of the event. If None, it is inferred from the shape of the parameter tensor.
        """
        if event_shape is None:
            event_shape = mean.shape
        super().__init__(event_shape=event_shape)

        validate_all_or_none(mean=mean, std=std)

        mean = init_parameter(param=mean, event_shape=event_shape, init=torch.randn)
        std = init_parameter(param=std, event_shape=event_shape, init=torch.rand)

        self.mean = nn.Parameter(mean)
        self.log_std = nn.Parameter(torch.empty_like(std))  # initialize empty, set with setter in next line
        self.std = std.clone().detach()

    @property
    def _supported_value(self):
        return 0.0

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Normal(self.mean, self.std)

    def params(self):
        return {"mean": self.mean, "std": self.std}

    @property
    def batch_shape(self):
        """Return batch shape of the distribution."""
        return self.distribution.batch_shape

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute weighted mean and standard deviation.

        Args:
            data: Input data tensor.
            weights: Weight tensor for each data point.
            bias_correction: Whether to apply bias correction to variance estimate.
        """
        n_total = weights.sum()
        mean_est = (weights * data).sum(0) / n_total

        centered = data - mean_est
        var_numerator = (weights * centered.pow(2)).sum(0)
        denom = n_total - 1 if bias_correction else n_total
        std_est = torch.sqrt(var_numerator / denom)

        # Handle edge cases (NaN, zero, or near-zero std) before broadcasting
        std_est = _handle_mle_edge_cases(std_est, lb=0.0)

        # Broadcast to event_shape and assign directly
        self.mean.data = self._broadcast_to_event_shape(mean_est)
        self.std = self._broadcast_to_event_shape(std_est)
