import torch
from torch import Tensor, nn

from spflow.modules.leaves.leaf import LeafModule
from spflow.utils.leaves import init_parameter, _handle_mle_edge_cases


class Laplace(LeafModule):
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
        num_repetitions: int = 1,
        parameter_fn: nn.Module = None,
        validate_args: bool | None = True,
        loc: Tensor = None,
        scale: Tensor = None,
    ):
        """Initialize Normal distribution.

        Args:
            scope: Variable scope (Scope, int, or list[int]).
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions (for 3D event shapes).
            parameter_fn: Optional neural network for parameter generation.
            loc: Mean tensor μ.
            scale: Standard deviation tensor σ > 0.
        """
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[loc, scale],
            parameter_fn=parameter_fn,
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
    def _torch_distribution_class(self) -> type[torch.distributions.Laplace]:
        return torch.distributions.Laplace

    def params(self):
        return {"loc": self.loc, "scale": self.scale}

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> dict[str, Tensor]:
        """Compute raw MLE estimates for normal distribution (without broadcasting).

        Args:
            data: Input data tensor.
            weights: Weight tensor for each data point.
            bias_correction: Whether to apply bias correction to variance estimate.

        Returns:
            Dictionary with 'loc' and 'scale' estimates (shape: out_features).
        """

        # Sum of weights
        n_total = weights.sum(dim=0)

        # ---- 1. Correct Laplace MLE for loc: weighted median ----
        loc_est = torch.median(data, dim=0).values

        # ---- 2. Correct Laplace MLE for scale ----
        abs_dev = torch.abs(data - loc_est)
        scale_est = (weights * abs_dev).sum(0) / n_total

        # Clean up edge cases
        scale_est = _handle_mle_edge_cases(scale_est, lb=0.0)

        return {"loc": loc_est, "scale": scale_est}

    def _set_mle_parameters(self, params_dict: dict[str, Tensor]) -> None:
        """Set MLE-estimated parameters for Normal distribution.

        Explicitly handles the two parameter types:
        - loc: Direct nn.Parameter, update .data attribute
        - scale: Property with setter, calls property setter which updates log_scale

        Args:
            params_dict: Dictionary with 'loc' and 'scale' parameter values.
        """
        self.loc.data = params_dict["loc"]
        self.scale = params_dict["scale"]

