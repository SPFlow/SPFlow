import torch
from torch import Tensor

from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule, parse_leaf_args


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
        r"""Initializes ``Uniform`` leaves node.

        Args:
            scope: Scope object specifying the scope of the distribution.
            out_channels: The number of output channels. If None, it is determined by the parameter tensors.
            num_repetitions: The number of repetitions for the leaves module.
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

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Uniform parameters are fixed buffers; nothing to update during MLE.

        Args:
            data: Scope-filtered data of shape (batch_size, num_scope_features).
            weights: Normalized weights of shape (batch_size, 1, ...).
            bias_correction: Not used for Uniform (fixed parameters).
        """
        pass

    def params(self) -> dict[str, Tensor]:
        return {"start": self.start, "end": self.end}
