"""
Created on July 08, 2021

@authors: Kevin Huy Nguyen

This file provides the inference methods for SPNs.
"""

import numpy as np
from numpy import ndarray
from scipy.special import logsumexp  # type: ignore
from spflow.base.structure.nodes.node import (
    Node,
    ISumNode,
    IProductNode,
    ILeafNode,
    eval_spn_bottom_up,
)
from spflow.base.structure.network_type import SPN
from .leaves.parametric import node_likelihood, node_log_likelihood
from typing import List, Callable, Type, Optional, Dict
from multipledispatch import dispatch  # type: ignore


def prod_log_likelihood(node: IProductNode, children: List[ndarray], **kwargs) -> ndarray:
    """
    Calculates the log-likelihood for a product node.

    Args:
        node:
            IProductNode to calculate log-likelihood for.
        children:
            np.array of child node values of IProductNode.

    Returns: Log-likelihood value for product node.
    """

    llchildren: ndarray = np.concatenate(children, axis=1)
    pll: ndarray = np.sum(llchildren, axis=1).reshape(-1, 1)
    pll[np.isinf(pll)] = np.finfo(pll.dtype).min

    return pll


def prod_likelihood(node: IProductNode, children: List[ndarray], **kwargs) -> ndarray:
    """
    Calculates the likelihood for a product node.

    Args:
        node:
            IProductNode to calculate likelihood for.
        children:
            np.array of child node values of IProductNode.

    Returns: likelihood value for product node.
    """

    llchildren: ndarray = np.concatenate(children, axis=1)

    return np.prod(llchildren, axis=1).reshape(-1, 1)


def sum_log_likelihood(node: ISumNode, children: List[ndarray], **kwargs) -> ndarray:
    """
    Calculates the log-likelihood for a sum node.

    Args:
        node:
            ISumNode to calculate log-likelihood for.
        children:
            np.array of child node values of ISumNode.

    Returns: Log-likelihood value for sum node.
    """

    llchildren: ndarray = np.concatenate(children, axis=1)
    b: ndarray = node.weights
    sll: ndarray = logsumexp(llchildren, b=b, axis=1).reshape(-1, 1)

    return sll


def sum_likelihood(node: ISumNode, children: List[ndarray], **kwargs) -> ndarray:
    """
    Calculates the likelihood for a sum node.

    Args:
        node:
            ISumNode to calculate likelihood for.
        children:
            np.array of child node values of ISumNode.

    Returns: Likelihood value for sum node.
    """

    llchildren: ndarray = np.concatenate(children, axis=1)
    b: ndarray = np.array(node.weights)

    return np.dot(llchildren, b).reshape(-1, 1)


_node_log_likelihood: Dict[Type, Callable] = {
    ISumNode: sum_log_likelihood,
    IProductNode: prod_log_likelihood,
    ILeafNode: node_log_likelihood,
}
_node_likelihood: Dict[Type, Callable] = {
    ISumNode: sum_likelihood,
    IProductNode: prod_likelihood,
    ILeafNode: node_likelihood,
}


@dispatch(SPN, Node, ndarray, node_likelihood=dict)
def likelihood(
    network_type: SPN,
    node: Node,
    data: ndarray,
    node_likelihood: Dict[Type, Callable] = _node_likelihood,
) -> ndarray:
    """
    Calculates the likelihood for a SPN.

    Args:
        network_type:
            Network Type to specify the inference method for SPNs.
        node:
            Root node of SPN to calculate likelihood for.
        data:
            Data given to evaluate ILeafNodes.
        node_likelihood:
            dictionary that contains k: Class of the node, v: lambda function that receives as parameters (node, args**)
            for leaf nodes and (node, [children results], args**) for other nodes.

    Returns: Likelihood value for SPN.
    """

    all_results: Optional[Dict[Node, ndarray]] = {}
    result: ndarray = eval_spn_bottom_up(node, node_likelihood, all_results=all_results, data=data)

    return result


@dispatch(SPN, Node, ndarray, node_log_likelihood=dict)
def log_likelihood(
    network_type: SPN,
    node: Node,
    data: ndarray,
    node_log_likelihood: Dict[Type, Callable] = _node_log_likelihood,
) -> ndarray:
    """
    Calculates the log-likelihood for a SPN.

    Args:
        network_type:
            Network Type to specify the inference method for SPNs.
        node:
            Root node of SPN to calculate log-likelihood for.
        data:
            Data given to evaluate ILeafNodes.
        node_log_likelihood:
            dictionary that contains k: Class of the node, v: lambda function that receives as parameters (node, args**)
            for leaf nodes and (node, [children results], args**) for other nodes.

    Returns: Log-likelihood value for SPN.
    """
    return likelihood(network_type, node, data, node_likelihood=node_log_likelihood)
