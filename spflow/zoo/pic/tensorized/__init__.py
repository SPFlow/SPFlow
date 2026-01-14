"""Tensorized layers for probabilistic circuits.

This module provides fused sum-product layers for efficient tensorized PC evaluation,
avoiding the K² explosion from explicit outer products.

Layers:
    - TensorizedLayer: Abstract base class
    - TuckerLayer: Tucker decomposition for arity-2 products
    - CollapsedCPLayer: CP decomposition with collapsed output matrix
    - UncollapsedCPLayer: CP decomposition with explicit rank
    - SharedCPLayer: CP with parameters shared across folds
    - MixingSumLayer: Pure sum layer for mixing
    - CPLayer: Factory function for CP variants

Utilities:
    - log_func_exp: Numerically stable log-space operations
"""

from spflow.zoo.pic.tensorized.base import TensorizedLayer
from spflow.zoo.pic.tensorized.cp import (
    CollapsedCPLayer,
    CPLayer,
    SharedCPLayer,
    UncollapsedCPLayer,
)
from spflow.zoo.pic.tensorized.mixing import MixingSumLayer
from spflow.zoo.pic.tensorized.tucker import TuckerLayer
from spflow.zoo.pic.tensorized.utils import log_func_exp

__all__ = [
    # Base
    "TensorizedLayer",
    # Tucker
    "TuckerLayer",
    # CP variants
    "CollapsedCPLayer",
    "UncollapsedCPLayer",
    "SharedCPLayer",
    "CPLayer",
    # Mixing
    "MixingSumLayer",
    # Utilities
    "log_func_exp",
]
