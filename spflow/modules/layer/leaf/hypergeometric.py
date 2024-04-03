#!/usr/bin/env python3

from torch import Tensor
from spflow.meta.data import Scope
from spflow import distributions as D
from spflow.modules.layer.leaf_layer import LeafLayer


class Hypergeometric(LeafLayer):
    def __init__(self, scope: Scope, K: Tensor, N: Tensor, n: Tensor):
        num_nodes_per_scope = 1 if K.dim() <= 1 else K.shape[1]
        super().__init__(scope, num_nodes_per_scope=num_nodes_per_scope)
        self.distribution = D.Hypergeometric(K, N, n)

    @property
    def device(self):
        return next(iter(self.buffers())).device