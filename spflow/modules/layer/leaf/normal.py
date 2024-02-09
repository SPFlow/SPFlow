#!/usr/bin/env python3

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import DispatchContext, init_default_dispatch_context
from typing import Optional, Union
from collections.abc import Iterable
from torch import Tensor
from spflow.meta.data import Scope
from spflow import distributions as D
from spflow.modules.layer.leaf_layer import LeafLayer


class Normal(LeafLayer):
    def __init__(self, scope: Scope, mean: Tensor = None, std: Tensor = None):
        num_nodes_per_scope = 1 if mean.dim() <= 1 else mean.shape[1]
        super().__init__(scope, num_nodes_per_scope=num_nodes_per_scope)
        self.distribution = D.Normal(mean, std)
