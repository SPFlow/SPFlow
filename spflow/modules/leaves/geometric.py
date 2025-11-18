import torch
from torch import Tensor, nn

from .distribution import Distribution
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import init_parameter, _handle_mle_edge_cases, parse_leaf_args
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class GeometricDistribution(Distribution):
    """Geometric distribution for modeling trials until first success.

    Parameterized by success probability p ∈ (0, 1] (stored in logit-space for numerical stability).
    """

    def __init__(self, p: Tensor = None, event_shape: tuple[int, ...] = None):
        """Initialize Geometric distribution.

        Args:
            p: Success probability tensor in (0, 1].
            event_shape: The shape of the event. If None, it is inferred from p shape.
        """
        if event_shape is None:
            event_shape = p.shape
        super().__init__(event_shape=event_shape)

        p = init_parameter(param=p, event_shape=event_shape, init=torch.rand)

        # Validate p at initialization
        if not ((p > 0) & (p <= 1)).all():
            raise ValueError("Success probability p must be in (0, 1]")
        if not torch.isfinite(p).all():
            raise ValueError("Success probability p must be finite")

        self.logit_p = nn.Parameter(proj_bounded_to_real(p, lb=0.0, ub=1.0))

    @property
    def p(self) -> Tensor:
        """Success probability in natural space (read via inverse projection of logit_p)."""
        return proj_real_to_bounded(self.logit_p, lb=0.0, ub=1.0)

    @p.setter
    def p(self, value: Tensor) -> None:
        """Set success probability (stores as logit_p, no validation after init)."""
        value_tensor = torch.as_tensor(value, dtype=self.logit_p.dtype, device=self.logit_p.device)
        self.logit_p.data = proj_bounded_to_real(value_tensor, lb=0.0, ub=1.0)

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 1

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying Geometric distribution."""
        return torch.distributions.Geometric(self.p)

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"p": self.p}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for success probability p.

        For Geometric distribution, the MLE is p = n / (sum(x_i) + n).

        Args:
            data: Scope-filtered data.
            weights: Normalized sample weights.
            bias_correction: Whether to apply bias correction.
        """
        n_total = weights.sum()
        n_success = (weights * data).sum(0)

        p_est = n_total / (n_success + n_total)
        if bias_correction:
            p_est = p_est - (p_est * (1 - p_est) / n_total)

        # Handle edge cases (NaN, zero, or near-zero p) before broadcasting
        p_est = _handle_mle_edge_cases(p_est, lb=0.0)

        # Broadcast to event_shape and assign - BoundedParameter ensures [0, 1]
        self.p = self._broadcast_to_event_shape(p_est)


class Geometric(LeafModule):
    """Geometric distribution leaf for modeling trials until first success.

    Parameterized by success probability p ∈ (0, 1].

    Attributes:
        p: Success probability (BoundedParameter).
        distribution: Underlying torch.distributions.Geometric.
    """

    def __init__(self, scope: Scope, out_channels: int = None, num_repetitions: int = None, p: Tensor = None):
        """Initialize Geometric distribution leaf.

        Args:
            scope: Variable scope.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions.
            p: Success probability in (0, 1].
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[p], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape
        self._distribution = GeometricDistribution(p=p, event_shape=event_shape)
