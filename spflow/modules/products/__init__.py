"""Product node implementations for probabilistic circuits.

This module provides various types of product nodes that implement factorization
operations in probabilistic circuits. Product nodes capture independence assumptions
between their child components.
"""

from .elementwise_product import ElementwiseProduct
from .outer_product import OuterProduct
from .product import Product
