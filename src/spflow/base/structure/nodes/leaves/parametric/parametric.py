"""
Created on June 11, 2021

@authors: Bennet Wittelsbach, Philipp Deibert
"""

from abc import ABC
from multipledispatch import dispatch  # type: ignore
from typing import List
from spflow.base.structure.nodes import ILeafNode, INode
from .statistical_types import ParametricType  # type: ignore

# TODO:
#   - Categorical
#   - ... ?


class ParametricLeaf(ILeafNode, ABC):
    """Base class for leaves that represent parametric probability distributions.

    Attributes:
        type:
            The parametric type of the distribution, either continuous or discrete

    """

    type: ParametricType

    def __init__(self, scope: List[int]) -> None:
        super().__init__(scope)


@dispatch(INode)  # type: ignore[no-redef]
def get_scipy_object(node: INode) -> None:
    """Get the associated scipy object of a parametric leaf node. This can be used to call the PDF, CDF, PPF, etc.

    The standard implementation accepts nodes of any type and raises an error, if there is no scipy
    object implemented for the given node. Else, the respective dispatched function will be called
    which returns the associated scipy object.

    Arguments:
        node:
            The node of which the respective scipy object shall be returned

    Returns:
        A scipy object representing the distribution of the given node, or None.

    Raises:
        NotImplementedError:
            The node is a ILeafNode and does not provide a scipy object or the node is not a ILeafNode
            and cannot provide a scipy object.

    """
    if type(node) is ILeafNode:
        raise NotImplementedError(f"{node} does not provide a scipy object")
    else:
        raise NotImplementedError(f"{node} cannot provide scipy objects")


@dispatch(INode)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: INode) -> None:
    """Get the parameters of a paremetric leaf node, s.t. they can be directly passed to the PDF, CDF, etc. of the
    associated scipy object.

    The standard implementation accepts nodes of any type and raises an error, if it is a leaf node that does
    not provide parameters or the node is not a leaf node. Else, the respective dispatched function will be called
    which returns the associated parameters.

    Arguments:
        node:
            The node of which the parameters shall be returned

    Returns:
        A dictionary with {"parameter": value}, or None.

    Raises:
        NotImplementedError:
            The node is a ILeafNode and does not provide parameters or the node is not a ILeafNode.
        InvalidParametersError:
            The parameters are None or set to invalid values.

    """
    if type(node) is ILeafNode:
        raise NotImplementedError(f"{node} does not provide any parameters")
    else:
        raise NotImplementedError(f"{node} cannot provide parameters")
