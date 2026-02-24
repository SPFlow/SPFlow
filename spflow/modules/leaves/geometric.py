import torch
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterCombinationError
from spflow.modules.leaves.leaf import LeafModule
from spflow.utils.leaves import init_parameter, _handle_mle_edge_cases
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded
from spflow.utils.sampling_context import SIMPLE


class Geometric(LeafModule):
    """Geometric distribution leaf for modeling trials until first success.

    Parameterized by success probability p ∈ (0, 1] (stored in logit-space for numerical stability).

    Attributes:
        p: Success probability (BoundedParameter).
        distribution: Underlying torch.distributions.Geometric.
    """

    def __init__(
        self,
        scope,
        out_channels: int = 1,
        num_repetitions=1,
        parameter_fn=None,
        validate_args: bool | None = True,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
    ):
        """Initialize Geometric distribution.

        Args:
            scope: Variable scope (Scope, int, or list[int]).
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions (for 3D event shapes).
            parameter_fn: Optional neural network for parameter generation.
            validate_args: Whether to enable torch.distributions argument validation.
            probs: Success probability tensor.
            logits: Log-odds tensor of the success probability.
        """
        if probs is not None and logits is not None:
            raise InvalidParameterCombinationError("Geometric accepts either probs or logits, not both.")

        param_source = logits if logits is not None else probs
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[param_source],
            parameter_fn=parameter_fn,
            validate_args=validate_args,
        )

        # Initialize parameters in well-behaved range to avoid extreme values
        def init_geometric_probs(shape):
            """Initialize probs in [0.1, 0.9] range to avoid MLE instability."""
            return torch.rand(shape) * 0.8 + 0.1

        init_fn = torch.randn if logits is not None else init_geometric_probs
        init_value = init_parameter(param=param_source, event_shape=self.event_shape, init=init_fn)

        logits_tensor = init_value if logits is not None else proj_bounded_to_real(init_value, lb=0.0, ub=1.0)

        self._logits = nn.Parameter(logits_tensor)

    @property
    def probs(self) -> Tensor:
        """Success probability in natural space (read via inverse projection of logits)."""
        return proj_real_to_bounded(self._logits, lb=0.0, ub=1.0)

    @probs.setter
    def probs(self, value: Tensor) -> None:
        """Set success probability (stores as logits)."""
        value_tensor = torch.as_tensor(value, dtype=self._logits.dtype, device=self._logits.device)
        self._logits.data = proj_bounded_to_real(value_tensor, lb=0.0, ub=1.0)

    @property
    def logits(self) -> Tensor:
        """Logits for the success probability."""
        return self._logits

    @logits.setter
    def logits(self, value: Tensor) -> None:
        value_tensor = torch.as_tensor(value, dtype=self._logits.dtype, device=self._logits.device)
        self._logits.data = value_tensor

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 1

    @property
    def _torch_distribution_class(self) -> type[torch.distributions.Geometric]:
        return torch.distributions.Geometric

    @property
    def _torch_distribution_class_with_differentiable_sampling(self) -> type[torch.distributions.Distribution]:
        return GeometricWithDifferentiableSamplingSIMPLE

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"logits": self.logits}

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> dict[str, Tensor]:
        """Compute raw MLE estimates for geometric distribution (without broadcasting).

        For Geometric distribution, the MLE is p = n / (sum(x_i) + n).

        Args:
            data: Input data tensor.
            weights: Weight tensor for each data point.
            bias_correction: Whether to apply bias correction.

        Returns:
            Dictionary with 'probs' estimate (shape: out_features).
        """
        n_total = weights.sum(dim=0)
        n_success = (weights * data).sum(0)

        p_est = n_total / (n_success + n_total)
        if bias_correction:
            p_est = p_est - (p_est * (1 - p_est) / n_total)

        # Handle edge cases (NaN, zero, or near-zero p) before broadcasting
        p_est = _handle_mle_edge_cases(p_est, lb=0.0)

        return {"probs": p_est}

    def _set_mle_parameters(self, params_dict: dict[str, Tensor]) -> None:
        """Set MLE-estimated parameters for Geometric distribution.

        Explicitly handles the parameter assignment:
        - probs: Property with setter, calls property setter which updates _logits

        Args:
            params_dict: Dictionary with 'probs' parameter value.
        """
        self.probs = params_dict["probs"]  # Uses property setter


class GeometricWithDifferentiableSamplingSIMPLE(torch.distributions.Geometric):
    """Geometric distribution with differentiable rsample via truncated SIMPLE.

    Notes:
        The Geometric distribution has infinite support over {0, 1, 2, ...}. This
        implementation uses a truncated support [0..Kmax] where Kmax is inferred
        from the current parameters and capped to keep computation bounded.
    """

    has_rsample = True
    _MAX_SUPPORT: int = 1024

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        sample_shape = torch.Size(sample_shape)

        probs = self.probs
        dtype = probs.dtype
        device = probs.device

        mean = (1.0 - probs) / probs
        std = torch.sqrt(1.0 - probs) / probs
        max_k = torch.ceil((mean + 10.0 * std + 10.0).max()).to(dtype=torch.int64)
        max_k_int = int(torch.clamp(max_k, min=0, max=self._MAX_SUPPORT).item())

        k = torch.arange(max_k_int + 1, device=device, dtype=dtype)  # (K,)
        value = k.reshape(max_k_int + 1, *([1] * len(self.batch_shape))).expand(max_k_int + 1, *self.batch_shape)

        base_dist = torch.distributions.Geometric(probs=probs, validate_args=False)
        logits = base_dist.log_prob(value).movedim(0, -1)
        if sample_shape:
            logits = logits.expand(*sample_shape, *logits.shape)

        samples_oh = SIMPLE(logits=logits, dim=-1, is_mpe=False)
        return (samples_oh * k).sum(dim=-1)
