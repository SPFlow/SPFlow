from torch import Tensor
from spflow.meta.data.scope import Scope
from spflow import distributions as D
from spflow.modules.node.leaf_node import LeafNode
from spflow.modules.layer.leaf.hypergeometric import Hypergeometric as HypergeometricLayer


class Hypergeometric(HypergeometricLayer):
    def __init__(self, scope: Scope, K: Tensor, N: Tensor, n: Tensor):
        super().__init__(scope=scope, K=K, N=N, n=n)
        assert K.shape == N.shape == n.shape == (1,), "M, N and n must be of shape (1,)"

    @property
    def device(self):
        return next(iter(self.buffers())).device