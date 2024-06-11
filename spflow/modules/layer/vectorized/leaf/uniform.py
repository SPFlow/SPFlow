from torch import Tensor
from spflow.meta.data.scope import Scope
from spflow import distributions as D
from spflow.modules.node.leaf_node import LeafNode
from spflow.modules.layer.leaf.uniform import Uniform as UniformLayer


class Uniform(UniformLayer):
    def __init__(self, scope: Scope, start: Tensor, end: Tensor, support_outside: Tensor = True):
        super().__init__(scope=scope, start=start, end=end, support_outside=support_outside)
        assert start.shape == end.shape == (1, start.shape[1]) or start.shape == end.shape == (start.shape[0], 1), "Mean and end must be of shape (1, n) or (n, 1)"

    @property
    def device(self):
        return next(iter(self.buffers())).device
