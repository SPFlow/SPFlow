from torch import Tensor
from spflow.meta.data.scope import Scope
from spflow import distributions as D
from spflow.modules.node.leaf_node import LeafNode
from spflow.modules.layer.leaf.log_normal import LogNormal as LogNormalLayer


class LogNormal(LogNormalLayer):
    def __init__(self, scope: Scope, mean: Tensor = None, std: Tensor = None):
        super().__init__(scope=scope, mean=mean, std=std)
        assert mean.shape == std.shape == (1,), "Mean and std must be of shape (1,)"
