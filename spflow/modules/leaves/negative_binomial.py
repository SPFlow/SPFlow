from torch import Tensor

from spflow.distributions.negative_binomial import NegativeBinomial as NegativeBinomialDistribution
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import parse_leaf_args


class NegativeBinomial(LeafModule):
    """Negative Binomial distribution leaf for modeling failures before r-th success.

    Note: Parameter n (number of successes) is fixed and cannot be learned.

    Attributes:
        n: Fixed number of required successes (buffer).
        p: Success probability in [0, 1] (BoundedParameter).
        distribution: Underlying torch.distributions.NegativeBinomial.
    """

    def __init__(
        self, scope: Scope, n: Tensor, out_channels: int = None, num_repetitions: int = None, p: Tensor = None
    ):
        """Initialize Negative Binomial distribution leaf.

        Args:
            scope: Variable scope.
            n: Fixed number of required successes (must be non-negative).
            out_channels: Number of output channels (inferred from p if None).
            num_repetitions: Number of repetitions.
            p: Success probability in [0, 1].
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[p], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape
        self._distribution = NegativeBinomialDistribution(n=n, p=p, event_shape=event_shape)
