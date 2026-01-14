"""Probabilistic Integral Circuits (PIC) module.

This module contains the PIC implementation including:
- Pipeline functions (rg2pic, pic2qpc)
- Integral module for latent variable integration
- Functional sharing utilities
- WeightedSum for quadrature integration
- Tensorized layers (Tucker, CP, etc.)
"""

from spflow.zoo.pic.functional_sharing import (
    FourierFeatures,
    FunctionGroup,
    MultiHeadedMLP,
    SharedMLP,
)
from spflow.zoo.pic.integral import Integral
from spflow.zoo.pic.pipeline import (
    MergeStrategy,
    PICInput,
    PICProduct,
    PICSum,
    QuadratureRule,
    pic2qpc,
    rg2pic,
)
from spflow.zoo.pic.tensorized.qpc import TensorizedQPC, TensorizedQPCConfig
from spflow.zoo.pic.weighted_sum import WeightedSum

__all__ = [
    # Pipeline
    "rg2pic",
    "pic2qpc",
    "MergeStrategy",
    "QuadratureRule",
    "PICInput",
    "PICSum",
    "PICProduct",
    "TensorizedQPC",
    "TensorizedQPCConfig",
    # Integral
    "Integral",
    # Functional sharing
    "FourierFeatures",
    "SharedMLP",
    "MultiHeadedMLP",
    "FunctionGroup",
    # Weighted sum
    "WeightedSum",
]
