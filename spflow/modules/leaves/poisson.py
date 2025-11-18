from torch import Tensor

from spflow.distributions.poisson import Poisson as PoissonDistribution
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import parse_leaf_args


class Poisson(LeafModule):
    """Poisson distribution leaf for modeling event counts.

    Parameterized by rate λ > 0 (stored in log-space).

    Attributes:
        rate: Rate parameter λ (LogSpaceParameter).
        distribution: Underlying torch.distributions.Poisson.
    """

    def __init__(
        self, scope: Scope, out_channels: int = None, num_repetitions: int = None, rate: Tensor = None
    ):
        """Initialize Poisson distribution leaf.

        Args:
            scope: Variable scope.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions.
            rate: Rate parameter λ > 0.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, num_repetitions=num_repetitions, params=[rate]
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape
        self._distribution = PoissonDistribution(rate=rate, event_shape=event_shape)
