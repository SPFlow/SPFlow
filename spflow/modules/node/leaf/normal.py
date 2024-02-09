from torch import Tensor
from spflow.meta.data.scope import Scope
from spflow import distributions as D
from spflow.modules.node.leaf_node import LeafNode
from spflow.modules.layer.leaf.normal import Normal as NormalLayer


# class Normal(LeafNode):
#     def __init__(self, scope: Scope, mean: Tensor = None, std: Tensor = None):
#         super().__init__(scope=scope)

#         assert mean.shape == std.shape == (1,), "Mean and std must be of shape (1,)"

#         self.distribution = D.Normal(mean, std)


class Normal(NormalLayer):
    def __init__(self, scope: Scope, mean: Tensor = None, std: Tensor = None):
        super().__init__(scope=scope, mean=mean, std=std)
        assert mean.shape == std.shape == (1,), "Mean and std must be of shape (1,)"
