from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor

from spflow.exceptions import InvalidParameterCombinationError, InvalidParameterError
from spflow.meta.data import Scope
from spflow.modules.leaves.leaf import LeafModule
from spflow.utils.cache import Cache


class Uniform(LeafModule):
    """Uniform distribution leaf with fixed interval bounds.

    Note: Interval bounds are fixed buffers and cannot be learned.

    Attributes:
        start: Start of interval (fixed buffer).
        end: End of interval (fixed buffer).
        end_next: Next representable value after end.
        support_outside: Whether values outside [start, end] are supported.
        distribution: Underlying torch.distributions.Uniform.
    """

    def __init__(
        self,
        scope: Scope,
        out_channels: int = 1,
        num_repetitions: int = 1,
        low: Tensor | None = None,
        high: Tensor | None = None,
        validate_args: bool | None = True,
        support_outside: bool = True,
    ):
        """Initialize Uniform distribution leaf.

        Args:
            scope: Variable scope.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions.
            low: Lower bound tensor (required).
            high: Upper bound tensor (required).
            validate_args: Whether to enable torch.distributions argument validation.
            support_outside: Whether values outside [start, end] are supported.
        """
        if low is None or high is None:
            raise InvalidParameterCombinationError(
                "'low' and 'high' parameters are required for Uniform distribution"
            )

        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise InvalidParameterError("Parameter must be finite")

        super().__init__(
            scope=scope,
            out_channels=out_channels,  # type: ignore
            num_repetitions=num_repetitions,
            params=[low, high],
            validate_args=validate_args,
        )

        # Register interval bounds as torch buffers (should not be changed)
        self.register_buffer("low", torch.empty(size=[]))
        self.register_buffer("high", torch.empty(size=[]))
        self.register_buffer("end_next", torch.empty(size=[]))
        self.register_buffer("_support_outside", torch.empty(size=[]))

        self.low = low
        self.high = high
        self.end_next = torch.nextafter(high, high.new_tensor(float("inf")))
        self._support_outside = high.new_tensor(support_outside, dtype=torch.bool)

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return self.low

    @property
    def _torch_distribution_class(self) -> type[torch.distributions.Uniform]:
        return torch.distributions.Uniform

    @property
    def mode(self) -> Tensor:
        """Returns the mode (midpoint) of the distribution."""
        return (self.low + self.high) / 2

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"low": self.low, "high": self.high}

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> dict[str, Tensor]:
        """Compute raw MLE estimates for uniform distribution (without broadcasting).

        Note: For Uniform distribution, this is a no-op since parameters are fixed buffers.
        This method exists to maintain consistency with other leaf distributions.

        Args:
            data: Input data tensor.
            weights: Weight tensor for each data point.
            bias_correction: Whether to apply bias correction (unused for Uniform).

        Returns:
            Empty dictionary (no parameters to estimate).
        """
        return {}

    def _set_mle_parameters(self, params_dict: dict[str, Tensor]) -> None:
        """Set MLE-estimated parameters for Uniform distribution.

        Note: For Uniform distribution, this is a no-op since parameters are fixed buffers.
        The low and high bounds cannot be updated through MLE.
        This method exists to maintain consistency with other leaf distributions.

        Args:
            params_dict: Dictionary with parameter values (empty for Uniform).
        """
        pass

    def _log_likelihood_interval(
        self,
        low: Tensor,
        high: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log P(low <= X <= high) for interval evidence.

        Args:
            low: Lower bounds of shape (batch, features). NaN = no lower bound.
            high: Upper bounds of shape (batch, features). NaN = no upper bound.
            cache: Optional cache dictionary.

        Returns:
            Log-likelihood tensor.
        """
        # Get scope-filtered bounds
        low_scoped = low[:, self.scope.query]
        high_scoped = high[:, self.scope.query]

        # Expand to match (batch, features, channels, repetitions)
        low_expanded = rearrange(low_scoped, "b f -> b f 1 1")
        high_expanded = rearrange(high_scoped, "b f -> b f 1 1")

        # Distribution bounds
        a = self.low  # (features, channels, reps) or scalar
        b = self.high

        # Handle NaN bounds (treat as -inf/+inf → clamp to distribution support)
        effective_low = torch.where(torch.isnan(low_expanded), a, torch.maximum(low_expanded, a))
        effective_high = torch.where(torch.isnan(high_expanded), b, torch.minimum(high_expanded, b))

        # Compute interval probability: (effective_high - effective_low) / (b - a)
        interval_length = torch.clamp(effective_high - effective_low, min=0.0)
        support_length = b - a

        prob = interval_length / support_length
        return torch.log(prob)

    def expectation_maximization(
        self,
        data: torch.Tensor,
        bias_correction: bool = False,
        cache: Cache | None = None,
    ) -> None:
        pass
