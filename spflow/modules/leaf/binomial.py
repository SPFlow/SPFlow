from torch import Tensor

from spflow import distributions as D
from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args


class Binomial(LeafModule):
    """
    Binomial distribution.
    """

    def __init__(
        self, scope: Scope, n: Tensor, out_channels: int = None, num_repetitions: int = None, p: Tensor = None
    ):
        """
        Args:
            scope (Scope): Scope of the module.
            n (Tensor): Number of trials.
            out_channels (int, optional): Number of output channels. If None, it will be inferred from `p`.
            num_repetitions (int, optional): Number of repetitions for the distribution.
            p (Tensor): Probability of success in each trial.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[p], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self.distribution = D.Binomial(n, p, event_shape=event_shape)
