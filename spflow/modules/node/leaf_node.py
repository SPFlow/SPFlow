"""Contains the basic abstract ``LeafNode`` module that all leaf nodes for SPFlow in the ``base`` backend.

All leaf nodes in the ``base`` backend should inherit from ``LeafNode`` or a subclass of it.
"""
from spflow.modules.node.leaf.utils import apply_nan_strategy
from spflow.modules.leaf_module import LeafModule
from spflow.meta.dispatch import SamplingContext
from torch import Tensor
from typing import Callable, Optional, Union
from spflow.meta.dispatch.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.sampling_context import init_default_sampling_context
from abc import ABC, abstractmethod
from collections.abc import Iterable
import torch
from spflow.meta.dispatch.dispatch import dispatch

from spflow.modules.node.node import Node
from spflow.meta.data.scope import Scope
from spflow.modules.layer.leaf_layer import LeafLayer


class LeafNode(LeafLayer, ABC):
    """Abstract base class for leaf nodes in the ``base`` backend.

    All valid SPFlow leaf nodes in the 'base' backend should inherit from this class or a subclass of it.

    Attributes:
        n_out:
            Integer indicating the number of outputs. One for nodes.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(self, scope: Scope) -> None:
        r"""Initializes ``LeafNode`` object.

        Args:
            scope:
                Scope object representing the scope of the leaf node,
        """
        assert len(scope.query) == 1, "Leaf nodes must have a scope with exactly one variable."
        super().__init__(scope=scope, num_nodes_per_scope=1)
