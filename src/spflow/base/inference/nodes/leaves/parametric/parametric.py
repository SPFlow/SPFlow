"""
Created on July 08, 2021

@authors: Kevin Huy Nguyen

This file provides the inference functions for the ILeafNodes.
"""

from multipledispatch import dispatch  # type: ignore
import numpy as np
from spflow.base.structure.nodes import INode

# TODO:
# marginalization
# typing data with np.ndarray doesnt work??
# Binomial, Negative Binomial?
POS_EPS = np.finfo(float).eps
MIN_NEG = np.finfo(float).min


@dispatch(INode, data=np.ndarray)  # type: ignore[no-redef]
def node_likelihood(node: INode, data: np.ndarray) -> np.ndarray:
    """Calculates the likelihood of node depending on the given data.

    The standard implementation accepts nodes of any type and raises an error, if it is a leaf node that does not have
    a likelihood calculation we support yet.

    Arguments:
        node:
            INode to calculate likelihood value of.
        data:
            Data given to evaluate INode.

    Returns:
        np.array with likelihood value for node.

    Raises:
        NotImplementedError:
            The node is a ILeafNode and does not provide parameters or the node is not a ILeafNode.
    """
    raise NotImplementedError(f"Likelihood not provided for {node}.")


@dispatch(INode, data=np.ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: INode, data: np.ndarray) -> np.ndarray:
    """Calculates the log-likelihood of node depending on the given data.

    The standard implementation accepts nodes of any type and raises an error, if it is a leaf node that does not have
    a log-ikelihood calculation we support yet.

    Arguments:
        node:
            INode to calculate log-likelihood value of.
        data:
            Data given to evaluate INode.

    Returns:
        np.array with log-likelihood value for node.

    Raises:
        NotImplementedError:
            The node is a ILeafNode and does not provide parameters or the node is not a ILeafNode.
    """
    raise NotImplementedError(f"Log-Likelihood not provided for {node}.")