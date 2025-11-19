import torch
from torch import Tensor, nn

from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import init_parameter, _handle_mle_edge_cases


class Normal(LeafModule):
    """Normal (Gaussian) distribution leaf module.

    Parameterized by mean μ and standard deviation σ (stored in log-space).

    Attributes:
        loc: Mean parameter.
        std: Standard deviation (accessed via property, stored as log_std).
    """

    def __init__(
        self,
        scope,
        out_channels: int = None,
        num_repetitions: int = None,
        parameter_network: nn.Module = None,
        validate_args: bool | None = True,
        loc: Tensor = None,
        scale: Tensor = None,
    ):
        """Initialize Normal distribution.

        Args:
            scope: Variable scope (Scope, int, or list[int]).
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions (for 3D event shapes).
            parameter_network: Optional neural network for parameter generation.
            loc: Mean tensor μ.
            scale: Standard deviation tensor σ > 0.
        """
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[loc, scale],
            parameter_network=parameter_network,
            validate_args=validate_args,
        )

        loc = init_parameter(param=loc, event_shape=self._event_shape, init=torch.randn)
        scale = init_parameter(param=scale, event_shape=self._event_shape, init=torch.rand)

        self.loc = nn.Parameter(loc)
        self.log_scale = nn.Parameter(torch.log(scale))

    @property
    def scale(self) -> Tensor:
        """Standard deviation in natural space (read via exp of log_std)."""
        return torch.exp(self.log_scale)

    @scale.setter
    def scale(self, value: Tensor) -> None:
        """Set standard deviation (stores as log_std, no validation after init)."""
        self.log_scale.data = torch.log(
            torch.as_tensor(value, dtype=self.log_scale.dtype, device=self.log_scale.device)
        )

    @property
    def _supported_value(self):
        return 0.0

    @property
    def _torch_distribution_class(self) -> type[torch.distributions.Normal]:
        return torch.distributions.Normal

    def params(self):
        return {"loc": self.loc, "scale": self.scale}

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
        loc_est = (weights * data).sum(0) / n_total

        centered = data - loc_est
        var_numerator = (weights * centered.pow(2)).sum(0)
        denom = n_total - 1 if bias_correction else n_total
        scale_est = torch.sqrt(var_numerator / denom)

        # Handle edge cases (NaN, zero, or near-zero std) before broadcasting
        scale_est = _handle_mle_edge_cases(scale_est, lb=0.0)

        # Broadcast to event_shape and assign directly
        self.loc.data = self._broadcast_to_event_shape(loc_est)
        self.scale = self._broadcast_to_event_shape(scale_est)
