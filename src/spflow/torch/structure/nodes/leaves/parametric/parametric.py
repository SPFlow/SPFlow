"""
Created on July 4, 2021
@authors: Philipp Deibert
"""

from abc import ABC
from typing import List, Union, Optional

from spflow.torch.structure.nodes.node import TorchLeafNode
from spflow.base.structure.nodes.leaves.parametric.statistical_types import ParametricType

import torch


def proj_real_to_bounded(
    x: torch.Tensor,
    lb: Optional[Union[float, torch.Tensor]] = None,
    ub: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    """Projects the real numbers onto a bounded interval."""
    if lb is not None and ub is not None:
        # project to bounded interval
        return torch.sigmoid(x) * (ub - lb) + lb
    elif ub is None:
        # project to left-bounded interval
        return torch.exp(x) + lb
    else:
        # project to right-bounded interval
        return -torch.exp(x) + ub


def proj_bounded_to_real(
    x: torch.Tensor,
    lb: Optional[Union[float, torch.Tensor]] = None,
    ub: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    """Projects a bounded interval onto the real numbers."""
    if lb is not None and ub is not None:
        # project from bounded interval
        return torch.log((x - lb) / (ub - x))
    elif ub is None:
        # project from left-bounded interval
        return torch.log(x - lb)
    else:
        # project from right-bounded interval
        return torch.log(ub - x)


class TorchParametricLeaf(TorchLeafNode, ABC):
    """Base class for Torch leaf nodes representing parametric probability distributions.
    Attributes:
        type (ParametricType): The parametric type of the distribution, either continuous or discrete.
    """

    ptype: ParametricType

    def __init__(self, scope: List[int]) -> None:
        super(TorchParametricLeaf, self).__init__(scope)
