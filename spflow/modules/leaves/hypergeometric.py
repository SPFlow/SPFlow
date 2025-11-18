from torch import Tensor

from spflow.distributions.hypergeometric import Hypergeometric as HypergeometricDistribution
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import parse_leaf_args


class Hypergeometric(LeafModule):
    """Hypergeometric distribution leaf for sampling without replacement.

    All parameters (K, N, n) are fixed buffers and cannot be learned.

    Attributes:
        K: Number of success states in population (fixed buffer).
        N: Population size (fixed buffer).
        n: Number of draws (fixed buffer).
        distribution: Underlying custom Hypergeometric distribution.
    """

    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        K: Tensor = None,
        N: Tensor = None,
        n: Tensor = None,
    ):
        """Initialize Hypergeometric distribution leaf module.

        Args:
            scope: Scope object specifying the scope of the distribution.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions for the distribution.
            K: Number of success states in population (fixed, non-negative).
            N: Population size (fixed, non-negative integer).
            n: Number of draws (fixed, non-negative integer).
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[K, N, n], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape
        self._distribution = HypergeometricDistribution(K=K, N=N, n=n, event_shape=event_shape)
