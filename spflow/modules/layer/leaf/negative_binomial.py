#!/usr/bin/env python3

from torch import Tensor
from spflow.meta.data import Scope
from spflow import distributions as D
from spflow.modules.layer.leaf_layer import LeafLayer


class NegativeBinomial(LeafLayer):
    def __init__(self, scope: Scope, n: Tensor, p: Tensor = None):
        num_nodes_per_scope = 1 if p.dim() <= 1 else p.shape[1]
        super().__init__(scope, num_nodes_per_scope=num_nodes_per_scope)
        self.distribution = D.NegativeBinomial(n, p)
