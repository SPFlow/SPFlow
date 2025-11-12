__version__ = "1.0.0"
__author__ = "The SPFlow Authors"

from .modules import (
    ElementwiseProduct,
    ElementwiseSum,
    Factorize,
    Module,
    OuterProduct,
    Product,
    Sum,
)
from .modules import ops
from .modules.rat import MixingLayer, RatSPN
from .modules.wrapper.image_wrapper import ImageWrapper
from .modules.wrapper.abstract_wrapper import AbstractWrapper
from .modules import leaves

__all__ = [
    "__version__",
    "__author__",
    "Module",
    "Sum",
    "Product",
    "Factorize",
    "OuterProduct",
    "ElementwiseProduct",
    "ElementwiseSum",
    "ops", # ops submodule, users can access ops via spflow.ops
    "RatSPN",
    "MixingLayer",
    "ImageWrapper",
    "AbstractWrapper",
    "leaves"  # leaves module, users can access leaves via spflow.leaves
]
