from torch import Tensor

from spflow import distributions as D
from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args


class Exponential(LeafModule):
    def __init__(self, scope: Scope, out_channels: int = None, rate: Tensor = None):
        """
        Initialize an Exponential distribution leaf module.

        Args:
            scope (Scope): The scope of the distribution.
            out_channels (int, optional): The number of output channels. If None, it is determined by the parameter tensor.
            rate (Tensor, optional): The rate parameter tensor.
        """
        event_shape = parse_leaf_args(scope=scope, out_channels=out_channels, params=[rate])
        super().__init__(scope, out_channels=event_shape[1])
        self.distribution = D.Exponential(rate, event_shape)
