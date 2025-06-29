from torch import Tensor

from spflow import distributions as D
from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args


class Gamma(LeafModule):
    def __init__(self, scope: Scope, out_channels: int = None, num_repetitions: int = None, alpha: Tensor = None, beta: Tensor = None):
        """
        Initialize a Gamma distribution leaf module.

        Args:
            scope (Scope): The scope of the distribution.
            out_channels (int, optional): The number of output channels. If None, it is determined by the parameter tensors.
            num_repetitions (int, optional): The number of repetitions for the leaf module.
            alpha (Tensor, optional): The alpha parameter tensor.
            beta (Tensor, optional): The beta parameter tensor.
        """
        event_shape = parse_leaf_args(scope=scope, out_channels=out_channels, params=[alpha, beta], num_repetitions=num_repetitions)
        super().__init__(scope, out_channels=event_shape[1])
        self.distribution = D.Gamma(alpha, beta, event_shape=event_shape)
