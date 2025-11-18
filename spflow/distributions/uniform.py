import torch
from torch import Tensor

from spflow.distributions.base import Distribution


class Uniform(Distribution):
    """Uniform distribution for continuous intervals with fixed bounds.

    Note: Interval bounds are fixed buffers and cannot be learned.

    Attributes:
        start: Start of interval (fixed buffer).
        end: End of interval (fixed buffer).
        end_next: Next representable value after end.
        support_outside: Whether values outside [start, end] are supported.
    """

    def __init__(
        self,
        start: Tensor = None,
        end: Tensor = None,
        support_outside: bool = True,
        event_shape: tuple[int, ...] = None,
    ):
        """Initialize Uniform distribution.

        Args:
            start: Start of interval (must be < end).
            end: End of interval (must be > start).
            support_outside: Whether values outside [start, end] are supported.
            event_shape: The shape of the event. If None, it is inferred from start/end shape.
        """
        if event_shape is None:
            event_shape = start.shape
        super().__init__(event_shape=event_shape)

        # Register interval bounds as torch buffers (should not be changed)
        self.register_buffer("start", torch.empty(size=[]))
        self.register_buffer("end", torch.empty(size=[]))
        self.register_buffer("end_next", torch.empty(size=[]))
        self.register_buffer("support_outside", torch.empty(size=[]))

        self.check_inputs(start, end, support_outside)

        self.start = start
        self.end = end
        self.end_next = torch.nextafter(end, torch.tensor(float("inf"), device=end.device))
        self.support_outside = torch.tensor(support_outside)

    def check_inputs(self, start: Tensor, end: Tensor, support_outside: bool):
        """Validate interval bounds."""
        if not torch.any(torch.isfinite(start)):
            raise ValueError(f"Values of 'start' for a uniform distribution must be finite, but was: {start}")
        if not torch.any(torch.isfinite(end)):
            raise ValueError(f"Values of 'end' for a uniform distribution must be finite, but was: {end}")
        if not (start < end).all():
            raise ValueError(f"Start must be smaller than end. Got start={start} and end={end}.")

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return self.start

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying Uniform distribution."""
        return torch.distributions.Uniform(self.start, self.end)

    @property
    def mode(self) -> Tensor:
        """Returns the mode (midpoint) of the distribution."""
        return (self.start + self.end) / 2

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"start": self.start, "end": self.end}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """No-op: Uniform parameters are fixed buffers and cannot be learned."""
        pass
