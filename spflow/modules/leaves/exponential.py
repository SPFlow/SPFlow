from torch import Tensor

from spflow.distributions.exponential import Exponential as ExponentialDistribution
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import parse_leaf_args


class Exponential(LeafModule):
    """Exponential distribution leaf for modeling time-between-events.

    Parameterized by rate λ > 0 (stored in log-space).

    Attributes:
        rate: Rate parameter λ (LogSpaceParameter).
        distribution: Underlying torch.distributions.Exponential.
    """

    def __init__(
        self, scope: Scope, out_channels: int = None, num_repetitions: int = None, rate: Tensor = None
    ):
        """Initialize Exponential distribution leaf.

        Args:
            scope: Variable scope.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions.
            rate: Rate parameter λ > 0.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[rate], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape
        self._distribution = ExponentialDistribution(rate=rate, event_shape=event_shape)
