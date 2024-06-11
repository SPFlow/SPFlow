from torch import Tensor
from spflow.meta.data.scope import Scope
from spflow import distributions as D
from spflow.modules.node.leaf_node import LeafNode
from spflow.modules.layer.leaf.categorical import Categorical as CategoricalLayer


class Categorical(CategoricalLayer):
    def __init__(self, scope: Scope, p: Tensor):
        super().__init__(scope=scope, p=p)
        assert p.shape[1] == 1 or p.shape[2]==1, "p must be of shape (k,)"
