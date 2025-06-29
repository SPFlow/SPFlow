from torch import Tensor

from spflow import distributions as D
from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args


class Hypergeometric(LeafModule):
    def __init__(
        self, scope: Scope, out_channels: int = None, num_repetitions: int = None, K: Tensor = None, N: Tensor = None, n: Tensor = None
    ):
        """
        Initialize a Hypergeometric distribution leaf module.

        Args:
            scope (Scope): The scope of the distribution.
            out_channels (int, optional): The number of output channels. If None, it is determined by the parameter tensors.
            num_repetitions (int, optional): The number of repetitions for the leaf module.
            K (Tensor, optional): The number of successes in the population.
            N (Tensor, optional): The population size.
            n (Tensor, optional): The number of draws.
        """
        event_shape = parse_leaf_args(scope=scope, out_channels=out_channels, params=[K, N, n], num_repetitions=num_repetitions)
        super().__init__(scope, out_channels=event_shape[1])
        self.distribution = D.Hypergeometric(K, N, n, event_shape=event_shape)

    @property
    def device(self):
        return next(iter(self.buffers())).device
