from torch import Tensor

from spflow import distributions as D
from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args


class Binomial(LeafModule):
    def __init__(self, scope: Scope, n: Tensor, out_channels: int = None, p: Tensor = None):
        event_shape = parse_leaf_args(scope=scope, out_channels=out_channels, params=[p])
        super().__init__(scope, out_channels=event_shape[1])
        self.distribution = D.Binomial(n, p, event_shape=event_shape)
