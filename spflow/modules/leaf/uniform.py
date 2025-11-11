import torch
from torch import Tensor

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule, MLEBatch
from spflow.utils.leaf import parse_leaf_args


class Uniform(LeafModule):
    # Interval bounds remain fixed buffers; descriptors are unnecessary here.
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

    def _mle_compute_statistics(self, batch: MLEBatch) -> dict[str, Tensor]:
        """Uniform parameters are fixed buffers; nothing to update during MLE."""
        return {}

    def _use_distribution_support(self) -> bool:
        """Uniform uses a half-open support; skip torch's closed interval."""
        return False

    def _custom_support_mask(self, data: Tensor) -> Tensor:
        """Allow start <= x < end, unless support_outside enables all values."""
        original_data_shape = data.shape
        start = self.start
        end = self.end

        # Add batch dimension to parameters
        start_expanded = start.unsqueeze(0)  # (1, features, channels) or (1, features, channels, repetitions)
        end_expanded = end.unsqueeze(0)

        # If data has fewer dimensions than start, expand it for the comparison
        # This handles the case where data is (batch, features) but start is (1, features, channels)
        data_expanded = data
        num_added_dims = 0
        while data_expanded.dim() < start_expanded.dim():
            data_expanded = data_expanded.unsqueeze(-1)  # Add trailing dimensions
            num_added_dims += 1

        interval_mask = (data_expanded >= start_expanded) & (data_expanded < end_expanded)
        mask = interval_mask | self.support_outside

        # Reduce the mask back to the original data shape
        # After broadcasting, the mask includes extra dimensions, so we take the first element along those
        for _ in range(num_added_dims):
            mask = mask[..., 0]  # Select first element along last dimension

        return mask

    def params(self) -> dict[str, Tensor]:
        return {"start": self.start, "end": self.end}
