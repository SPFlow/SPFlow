import torch
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterCombinationError
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import init_parameter, _handle_mle_edge_cases
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded


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
        out_channels=None,
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

        init_fn = torch.randn if logits is not None else torch.rand
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
        n_total = weights.sum()
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

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for success probability p.

        For Geometric distribution, the MLE is p = n / (sum(x_i) + n).

        Args:
            data: Scope-filtered data.
            weights: Normalized sample weights.
            bias_correction: Whether to apply bias correction.
        """
        estimates = self._compute_parameter_estimates(data, weights, bias_correction)

        # Broadcast to event_shape and assign via property setter
        self.probs = self._broadcast_to_event_shape(estimates["probs"])
