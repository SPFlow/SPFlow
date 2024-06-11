from torch import Tensor
from spflow.meta.data.scope import Scope

from spflow.modules.layer.leaf.normal import Normal as NormalLayer

class Normal(NormalLayer):
    def __init__(self, scope: Scope, mean: Tensor = None, std: Tensor = None):
        super().__init__(scope=scope, mean=mean, std=std)
        assert mean.shape == std.shape == (1, mean.shape[1]) or mean.shape == std.shape == (mean.shape[0], 1), "Mean and std must be of shape (1, n) or (n, 1)"