from torch import Tensor
from spflow.meta.data.scope import Scope
from spflow import distributions as D
from spflow.modules.node.leaf_node import LeafNode
from spflow.modules.layer.leaf.poisson import Poisson as PoissonLayer

class Poisson(PoissonLayer):
    def __init__(self, scope: Scope, rate: Tensor = None):
        super().__init__(scope=scope, rate=rate)
        assert rate.shape == (1, rate.shape[1]) or rate.shape == (rate.shape[0], 1), "Rate must be of shape (1, n) or (n, 1)"