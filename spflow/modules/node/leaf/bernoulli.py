from torch import Tensor
from spflow.meta.data.scope import Scope
from spflow import distributions as D
from spflow.modules.node.leaf_node import LeafNode
from spflow.modules.layer.leaf.bernoulli import Bernoulli as BernoulliLayer


class Bernoulli(BernoulliLayer):
    def __init__(self, scope: Scope, p: Tensor = None):
        super().__init__(scope=scope, p=p)
        assert p.shape == (1,), "p must be of shape (1,)"