from torch import Tensor

from spflow import distributions as D
from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args


class LogNormal(LeafModule):
    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        mean: Tensor = None,
        std: Tensor = None,
    ):
        """
        Initialize a LogNormal distribution leaf module.

        Args:
            scope (Scope): The scope of the distribution.
            out_channels (int, optional): The number of output channels. If None, it is determined by the parameter tensors.
            num_repetitions (int, optional): The number of repetitions for the leaf module.
            mean (Tensor, optional): The mean parameter tensor.
            std (Tensor, optional): The standard deviation parameter tensor.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[mean, std], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self.distribution = D.LogNormal(mean, std, event_shape=event_shape)
