from torch import Tensor

from spflow.distributions.log_normal import LogNormal as LogNormalDistribution
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import parse_leaf_args


class LogNormal(LeafModule):
    """Log-Normal distribution leaf for modeling positive-valued data.

    Note: Parameters μ and σ apply to ln(x), not x itself.

    Attributes:
        mean: Mean μ of log-space distribution.
        std: Standard deviation σ > 0 of log-space distribution (LogSpaceParameter).
        distribution: Underlying torch.distributions.LogNormal.
    """

    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        mean: Tensor = None,
        std: Tensor = None,
    ):
        """Initialize Log-Normal distribution leaf.

        Args:
            scope: Variable scope.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions.
            mean: Mean μ of log-space distribution.
            std: Standard deviation σ > 0 of log-space distribution.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[mean, std], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape
        self._distribution = LogNormalDistribution(mean=mean, std=std, event_shape=event_shape)

    @property
    def mean(self):
        """Delegate to distribution's mean."""
        return self._distribution.mean

    @property
    def std(self):
        """Delegate to distribution's std."""
        return self._distribution.std
