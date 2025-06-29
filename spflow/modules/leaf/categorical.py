from torch import Tensor

from spflow import distributions as D
from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args


class Categorical(LeafModule):
    def __init__(self, scope: Scope, out_channels: int = None, num_repetitions: int = None, K: int = None, p: Tensor = None):
        """
        Initialize a Categorical distribution leaf module.

        Args:
            scope (Scope): The scope of the distribution.
            out_channels (int, optional): The number of output channels. If None, it is determined by the parameter tensor.
            num_repetitions (int, optional): The number of repetitions for the leaf module.
            K (int, optional): The number of categories.
            p (Tensor, optional): The probability tensor.
        """
        event_shape = parse_leaf_args(scope=scope, out_channels=out_channels, params=[p], num_repetitions=num_repetitions)
        super().__init__(scope, out_channels=event_shape[1])
        self.distribution = D.Categorical(p, K=K, event_shape=event_shape)
