"""Data structures for metadata management in probabilistic circuits.

This module provides core data structures for managing metadata, including
variable scopes, feature types, and contextual information needed for
proper circuit construction and validation. These structures ensure type safety and proper variable handling
throughout the SPFlow ecosystem.
"""

from .feature_context import FeatureContext
from .feature_types import (
    BernoulliType,
    BinomialType,
    ExponentialType,
    FeatureType,
    FeatureTypes,
    GammaType,
    NormalType,
    GeometricType,
    HypergeometricType,
    LogNormalType,
    NegativeBinomialType,
    PoissonType,
    UniformType,
)
from .meta_type import MetaType
from .scope import Scope
