from torch import Tensor

from spflow.distributions.gamma import Gamma as GammaDistribution
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import parse_leaf_args


class Gamma(LeafModule):
    """Gamma distribution leaf for modeling positive-valued continuous data.

    Parameterized by shape α > 0 and rate β > 0 (both stored in log-space).

    Attributes:
        alpha: Shape parameter α (LogSpaceParameter).
        beta: Rate parameter β (LogSpaceParameter).
        distribution: Underlying torch.distributions.Gamma.
    """

    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        alpha: Tensor = None,
        beta: Tensor = None,
    ):
        """Initialize Gamma distribution leaf.

        Args:
            scope: Variable scope.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions.
            alpha: Shape parameter α > 0.
            beta: Rate parameter β > 0.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[alpha, beta], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape
        self._distribution = GammaDistribution(alpha=alpha, beta=beta, event_shape=event_shape)
