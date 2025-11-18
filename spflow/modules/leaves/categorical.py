import torch
from torch import Tensor

from spflow.distributions.categorical import Categorical as CategoricalDistribution
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import parse_leaf_args


class Categorical(LeafModule):
    """Categorical distribution leaf for discrete choice over K categories.

    Attributes:
        p: Categorical probabilities (normalized, includes extra dimension for K).
        K: Number of categories.
        distribution: Underlying torch.distributions.Categorical.
    """

    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        K: int = None,
        p: Tensor = None,
    ):
        """Initialize Categorical distribution leaf module.

        Args:
            scope: The scope of the distribution.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions for the distribution.
            K: Number of categories.
            p: Probability tensor of shape (*event_shape, K).
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[p], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        # Normalize probabilities if provided to ensure they sum to 1 along last dimension
        # Only normalize if probabilities are valid (non-negative, <= 1.0, and finite) but don't sum to 1
        if p is not None:
            if torch.all(p >= 0.0) and torch.all(p <= 1.0) and torch.isfinite(p).all():
                # Check if probabilities already sum to 1 (within tolerance)
                sums = p.sum(dim=-1, keepdim=True)
                if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6):
                    p = p / sums
            # If probabilities are invalid, let SimplexParameter handle validation

        self._distribution = CategoricalDistribution(p=p, K=K, event_shape=event_shape)
