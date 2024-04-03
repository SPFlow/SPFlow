#!/usr/bin/env python3

from torch import Tensor
from spflow.meta.data import Scope
from spflow import distributions as D
from spflow.modules.layer.leaf_layer import LeafLayer


class Uniform(LeafLayer):
    def __init__(self, scope: Scope, start: Tensor, end: Tensor, support_outside: Tensor = True):
        num_nodes_per_scope = 1 if start.dim() <= 1 else start.shape[1]
        super().__init__(scope, num_nodes_per_scope=num_nodes_per_scope)
        self.distribution = D.Uniform(start, end, support_outside)

    @property
    def device(self):
        return next(iter(self.buffers())).device
