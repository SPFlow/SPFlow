from torch import Tensor
from spflow.meta.data.scope import Scope
from spflow import distributions as D
from spflow.modules.node.leaf_node import LeafNode
from spflow.modules.layer.leaf.exponential import Exponential as ExponentialLayer


class Exponential(ExponentialLayer):
    def __init__(self, scope: Scope, rate: Tensor = None):
        super().__init__(scope=scope, rate=rate)
        assert (isinstance(rate, float) or rate.shape == (1,)), "rate must be of shape (1,) or float"