from torch import Tensor

from spflow import distributions as D
from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args


class Bernoulli(LeafModule):
    r"""
    Create a Bernoulli leaf module.
    """
    def __init__(self, scope: Scope, out_channels: int = None, num_repetitions: int = None, p: Tensor = None):
        r"""
        Args:
            scope (Scope): The scope of the leaf module.
            out_channels (int, optional): The number of output channels. If None, it is inferred from the shape of the parameter tensor.
            num_repetitions (int, optional): The number of repetitions for the leaf module.
            p (Tensor): PyTorch tensor representing the success probabilities of the Bernoulli distributions.
        """
        event_shape = parse_leaf_args(scope=scope, out_channels=out_channels, params=[p], num_repetitions=num_repetitions)
        super().__init__(scope, out_channels=event_shape[1])
        self.distribution = D.Bernoulli(p, event_shape=event_shape)
