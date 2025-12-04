from __future__ import annotations

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
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
        out_channels: int = None,
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

        super().__init__(
            scope=scope,
            out_channels=out_channels,
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
        self.end_next = torch.nextafter(high, torch.tensor(float("inf"), device=high.device))
        self._support_outside = torch.tensor(support_outside)

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

    def expectation_maximization(
            self,
            data: torch.Tensor,
            bias_correction: bool = False,
            cache: Cache | None = None,
    ) -> None:
        pass
