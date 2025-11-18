from torch import Tensor

from spflow.distributions.geometric import Geometric as GeometricDistribution
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import parse_leaf_args


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
