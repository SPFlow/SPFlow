import torch
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta.data import Scope
from spflow.modules.leaves.leaf import LeafModule
from spflow.utils.leaves import init_parameter
from spflow.utils.projections import proj_convex_to_real, proj_real_to_convex


class Categorical(LeafModule):
    """Categorical distribution leaf for discrete choice over K categories.

    Attributes:
        p: Categorical probabilities (normalized, includes extra dimension for K).
        K: Number of categories.
        distribution: Underlying torch.distributions.Categorical.
    """

    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = 1,
        K: int | Tensor | None = None,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        parameter_fn: nn.Module = None,
        validate_args: bool | None = True,
    ):
        """Initialize Categorical distribution leaf module.

        Args:
            scope: The scope of the distribution.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions for the distribution.
            K: Number of categories (optional if parameter tensor provided).
            probs: Probability tensor over categories.
            logits: Logits tensor over categories.
            parameter_fn: Optional neural network for parameter generation.
            validate_args: Whether to enable torch.distributions argument validation.
        """
        # K can be inferred from provided tensor if available
        if K is None and probs is None and logits is None:
            raise InvalidParameterCombinationError(
                "Either 'K' or one of probs/logits must be provided for Categorical distribution"
            )

        param_source = logits if logits is not None else probs

        if K is None and param_source is not None:
            # Infer K from the last dimension of p
            K = param_source.shape[-1]

        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[param_source],
            parameter_fn=parameter_fn,
            validate_args=validate_args,
        )
        self.K = K

        # Initialize parameter with K categories
        param_shape = (*self.event_shape, K)
        init_value = init_parameter(
            param=param_source,
            event_shape=param_shape,
            init=lambda shape: torch.rand(shape).softmax(dim=-1),
        )

        logits_tensor = init_value if logits is not None else proj_convex_to_real(init_value)

        self._logits = nn.Parameter(logits_tensor)

    @property
    def probs(self) -> Tensor:
        """Categorical probabilities in natural space (read via softmax of logits)."""
        return proj_real_to_convex(self._logits)

    @probs.setter
    def probs(self, value: Tensor) -> None:
        """Set categorical probabilities (stores as logits)."""
        value_tensor = torch.as_tensor(value, dtype=self._logits.dtype, device=self._logits.device)
        self._logits.data = proj_convex_to_real(value_tensor)

    @property
    def logits(self) -> Tensor:
        """Logits directly parameterizing the categorical distribution."""
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
    def _torch_distribution_class(self) -> type[torch.distributions.Categorical]:
        return torch.distributions.Categorical

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"logits": self.logits}

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> dict[str, Tensor]:
        """Compute raw MLE estimates for categorical distribution (without broadcasting).

        Args:
            data: Input data tensor.
            weights: Weight tensor for each data point.
            bias_correction: Not used for Categorical (included for interface consistency).

        Returns:
            Dictionary with 'probs' estimates (shape: out_features x K).
        """
        n_total = weights.sum(dim=0)

        if self.K is not None:
            num_categories = self.K
        else:
            finite_values = data[~torch.isnan(data)]
            num_categories = int(finite_values.max().item()) + 1 if finite_values.numel() else 1

        p_est = torch.empty_like(self.probs)
        for cat in range(num_categories):
            cat_mask = (data == cat).float()
            p_est[..., cat] = torch.sum(weights * cat_mask, dim=0) / n_total

        # Handle edge cases (NaN or invalid probabilities) before broadcasting
        # For categorical, we ensure probabilities sum to 1 and are non-negative
        p_est = torch.clamp(p_est, min=1e-10)  # Avoid zero probabilities
        p_est = p_est / p_est.sum(dim=-1, keepdim=True)  # Renormalize
        # TODO: check if this is the correct dim ^

        return {"probs": p_est}

    def _set_mle_parameters(self, params_dict: dict[str, Tensor]) -> None:
        """Set MLE-estimated parameters for Categorical distribution.

        Explicitly handles the parameter type:
        - probs: Property with setter, calls property setter which updates _logits

        Args:
            params_dict: Dictionary with 'probs' parameter values.
        """
        self.probs = params_dict["probs"]  # Uses property setter
