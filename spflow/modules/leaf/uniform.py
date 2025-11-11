import torch
from torch import Tensor
from typing import Optional, Callable

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args
from spflow.utils.cache import Cache


class Uniform(LeafModule):
    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        start: Tensor = None,
        end: Tensor = None,
        support_outside: bool = True,
    ):
        r"""Initializes ``Uniform`` leaf node.

        Args:
            scope: Scope object specifying the scope of the distribution.
            out_channels: The number of output channels. If None, it is determined by the parameter tensors.
            num_repetitions: The number of repetitions for the leaf module.
            start: PyTorch tensor containing the start of the intervals (including).
            end: PyTorch tensor containing the end of the intervals (including). Must be larger than 'start'.
            support_outside:
                PyTorch tensor containing booleans indicating whether or not values outside of the intervals are part of the support.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[start, end], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        # register interval bounds as torch buffers (should not be changed)
        self.register_buffer("start", torch.empty(size=[]))
        self.register_buffer("end", torch.empty(size=[]))
        self.register_buffer("end_next", torch.empty(size=[]))
        self.register_buffer("support_outside", torch.empty(size=[]))

        self.check_inputs(start, end, support_outside)

        self.start = start
        self.end = end
        self.end_next = torch.nextafter(end, torch.tensor(float("inf"), device=end.device))
        self.support_outside = torch.tensor(support_outside)

    def check_inputs(self, start: Tensor, end: Tensor, support_outside: torch.Tensor):
        if not torch.any(torch.isfinite(start)):
            raise ValueError(f"Values of 'start' for a uniform distribution must be finite, but was: {start}")
        if not torch.any(torch.isfinite(end)):
            raise ValueError(f"Values of 'end' for a uniform distribution must be finite, but was: {end}")
        if not (start < end).all():
            raise ValueError(f"Start must be smaller than end. Got start={start} and end={end}.")

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Uniform(self.start, self.end)

    def mode(self):
        return (self.start + self.end) / 2

    @property
    def _supported_value(self):
        return self.start

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Optional[Tensor] = None,
        bias_correction: bool = True,
        nan_strategy: Optional[str | Callable] = None,
        check_support: bool = True,
        cache: Cache | None = None,
        preprocess_data: bool = True,
    ) -> None:
        """
        All parameters of the Uniform distribution are regarded as fixed and will not be estimated.
        Therefore, this method does nothing, but check for the validity of the data.

        Args:
            data: The input data tensor.
            weights: Optional weights tensor. If None, uniform weights are created.
            bias_correction: Unused for Uniform, kept for API consistency.
            nan_strategy: Optional string or callable specifying how to handle missing data.
            check_support: Boolean value indicating whether to check data support.
            cache: Optional cache dictionary.
            preprocess_data: Boolean indicating whether to select relevant data for scope.
        """
        # Always select data relevant to this scope (same as log_likelihood does)
        data = data[:, self.scope.query]
        data = data.unsqueeze(2)
        if torch.any(~self.check_support(data)):
            raise ValueError("Encountered values outside of the support for uniform distribution.")

        # do nothing since there are no learnable parameters
        pass

    def check_support(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions."""

        # torch distribution support is an interval, despite representing a distribution over a half-open interval
        # end is adjusted to the next largest number to make sure that desired end is part of the distribution interval
        # may cause issues with the support check; easier to do a manual check instead

        if self.num_repetitions is not None and data.dim() < 4:
            data = data.unsqueeze(-1)

        valid = torch.ones_like(data, dtype=torch.bool)

        # check if values are within valid range
        # check only first entry of num_leaf node dim since all leaf node repetition have the same support
        if self.num_repetitions is not None:
            valid &= ((data >= self.start) & (data < self.end))[..., [0], :1]
        else:
            valid &= ((data >= self.start) & (data < self.end))[..., [0]]
        valid |= self.support_outside

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(data)
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~(data[~nan_mask & valid].isinf())

        return valid

    def params(self):
        return {"start": self.start, "end": self.end}

    @property
    def device(self):
        """
        Get the device of the module. Necessary hack since this module has no parameters.

        Returns:
            torch.device: The device on which the module's buffers are located.
        """
        return next(iter(self.buffers())).device
