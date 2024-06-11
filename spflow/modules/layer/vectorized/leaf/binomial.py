from torch import Tensor
from spflow.meta.data.scope import Scope
from spflow import distributions as D
from spflow.modules.node.leaf_node import LeafNode
from spflow.modules.layer.leaf.binomial import Binomial as BinomialLayer


class Binomial(BinomialLayer):
    def __init__(self, scope: Scope,  n: Tensor, p: Tensor = None):
        super().__init__(scope=scope, n=n, p=p)
        assert p.shape == (1, p.shape[1]) or p.shape == (p.shape[0], 1), "P must be of shape (1, n) or (n, 1)"