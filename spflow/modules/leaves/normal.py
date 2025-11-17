from torch import Tensor

from spflow.distributions.normal import Normal as NormalDistribution
from spflow.meta.data import Scope
from spflow.modules.leaves.base import (
    LeafModule,
)
from spflow.utils.leaves import LogSpaceParameter, parse_leaf_args


class Normal(LeafModule):
    """Normal (Gaussian) distribution leaf module.

    Parameterized by mean μ and standard deviation σ (stored in log-space).

    Attributes:
        mean: Mean parameter.
        std: Standard deviation (LogSpaceParameter).
        distribution: Underlying torch.distributions.Normal.
    """

    std = LogSpaceParameter("std")

    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        mean: Tensor = None,
        std: Tensor = None,
    ):
        """Initialize Normal distribution leaf.

        Args:
            scope: Variable scope.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions.
            mean: Mean parameter tensor (random init if None).
            std: Standard deviation tensor (must be positive, random init if None).
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, num_repetitions=num_repetitions, params=[mean, std]
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape
        self._distribution = NormalDistribution(mean=mean, std=std, event_shape=event_shape)
