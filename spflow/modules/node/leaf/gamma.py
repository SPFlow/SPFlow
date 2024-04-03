from torch import Tensor
from spflow.meta.data.scope import Scope
from spflow import distributions as D
from spflow.modules.node.leaf_node import LeafNode
from spflow.modules.layer.leaf.gamma import Gamma as GammaLayer


class Gamma(GammaLayer):
    def __init__(self, scope: Scope, alpha: Tensor = None, beta: Tensor = None):
        super().__init__(scope=scope, alpha=alpha, beta=beta)
        assert alpha.shape == beta.shape == (1,), "Alpha and beta must be of shape (1,)"

