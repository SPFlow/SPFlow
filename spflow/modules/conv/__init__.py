"""Convolutional probabilistic circuit modules.

Provides convolutional sum and product layers for modeling spatial structure
in image data with probabilistic circuits.
"""

from spflow.modules.conv.conv_pc import ConvPc
from spflow.modules.conv.prod_conv import ProdConv
from spflow.modules.conv.sum_conv import SumConv

__all__ = ["SumConv", "ProdConv", "ConvPc"]
