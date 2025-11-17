import torch
from torch import Tensor

from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from utils.leaves import parse_leaf_args


class Uniform(LeafModule):
    """Uniform distribution leaf with fixed interval bounds.

    Note: Interval bounds are fixed buffers and cannot be learned.

    Attributes:
        start (Tensor): Start of interval (buffer).
        end (Tensor): End of interval (buffer).
        support_outside (Tensor): Whether values outside [start, end] are supported.
        distribution: Underlying torch.distributions.Uniform object.
    """

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
        """Initialize Uniform distribution leaf.

        Args:
            scope: Variable scope for this distribution.
            out_channels: Number of output channels.
            num_repetitions: Number of repetitions.
            start: Start of interval (must be < end).
            end: End of interval (must be > start).
            support_outside: Whether values outside [start, end] are supported.
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

    def mode(self) -> Tensor:
        """Returns the mode (midpoint) of the distribution."""
        return (self.start + self.end) / 2

    @property
    def _supported_value(self):
        return self.start

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """No-op: Uniform parameters are fixed buffers."""
        pass

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"start": self.start, "end": self.end}
