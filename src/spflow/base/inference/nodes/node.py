"""
Created on July 08, 2021

@authors: Kevin Huy Nguyen

This file provides the inference methods for SPNs.
"""
import numpy as np
from numpy import ndarray
from scipy.special import logsumexp  # type: ignore

from spflow.base.structure.nodes.node import (
    ISumNode,
    IProductNode,
    ILeafNode,
)

from spflow.base.structure.network_type import SPN, NetworkType
from .leaves.parametric import node_log_likelihood
from typing import List, Callable, Type, Dict
from multipledispatch import dispatch  # type: ignore
from spflow.base.memoize import memoize
from spflow.base.inference.nodes.leaves.parametric import node_likelihood


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


@dispatch(ISumNode, np.ndarray, SPN, node_log_likelihood_dict=dict, cache=dict)
@memoize(ISumNode)
def log_likelihood(
    node: ISumNode,
    data: np.ndarray,
    network_type: NetworkType,
    node_log_likelihood_dict=None,
    cache: Dict = {},
) -> np.ndarray:
    """
    Recursively calculates the log_likelihood for a ISumNode.
    It calls log_likelihood on all it children to calculate its own value by using the fitting evaluation function.

    Args:
        node:
            ISumNode to calculate log_likelihood for.
        data:
            Given observed or missing data.
        network_type:
            NetworkType to dispatch to this evaluation method. Expected to be SPN.
        node_log_likelihood_dict:
            Dictionary containing evaluation methods for SPNs.
        cache:
            Dictionary collecting which nodes were already calculated to avoid recalculating them.

    Returns: Log_likelihood value for ISumNode.
    """
    node_log_likelihood_dict = node_log_likelihood_dict or _node_log_likelihood
    inputs = [
        log_likelihood(
            child, data, SPN(), node_log_likelihood_dict=_node_log_likelihood, cache=cache
        )
        for child in node.children
    ]
    eval_func = node_log_likelihood_dict[type(node)]
    return eval_func(node, inputs)


@dispatch(IProductNode, np.ndarray, SPN, node_log_likelihood_dict=dict, cache=dict)
@memoize(IProductNode)
def log_likelihood(
    node: IProductNode,
    data: np.ndarray,
    network_type: NetworkType,
    node_log_likelihood_dict=None,
    cache: Dict = {},
) -> np.ndarray:
    """
    Recursively calculates the log_likelihood for a IProdNode.
    It calls log_likelihood on all it children to calculate its own value by using the fitting evaluation function.

    Args:
        node:
            IProdNode to calculate log_likelihood for.
        data:
            Given observed or missing data.
        network_type:
            NetworkType to dispatch to this evaluation method. Expected to be SPN.
        node_log_likelihood_dict:
            Dictionary containing evaluation methods for SPNs.
        cache:
            Dictionary collecting which nodes were already calculated to avoid recalculating them.

    Returns: Log_likelihood value for IProdNode.
    """
    node_log_likelihood_dict = node_log_likelihood_dict or _node_log_likelihood
    inputs = [
        log_likelihood(
            child, data, SPN(), node_log_likelihood_dict=_node_log_likelihood, cache=cache
        )
        for child in node.children
    ]
    eval_func = node_log_likelihood_dict[type(node)]
    return eval_func(node, inputs)


@dispatch(ILeafNode, np.ndarray, SPN, node_log_likelihood_dict=dict, cache=dict)
@memoize(ILeafNode)
def log_likelihood(
    node: ILeafNode,
    data: np.ndarray,
    network_type: NetworkType,
    node_log_likelihood_dict=None,
    cache: Dict = {},
) -> np.ndarray:
    """
    Calculates log_likelihood for a ILeafNode by evaluating the distribution represented by the leaf node type.

    Args:
        node:
            ILeafNode node to calculate log_likelihood for.
        data:
            Given observed or missing data.
        network_type:
            NetworkType to dispatch to this evaluation method. Expected to be SPN.
        node_log_likelihood_dict:
            Only used to dispatch to this function.
        cache:
            Dictionary collecting which nodes were already calculated to avoid recalculating them.

    Returns: Log_likelihood value for ILeafNode.
    """
    return node_log_likelihood(node, data=data)


@dispatch(ISumNode, np.ndarray, SPN, node_likelihood_dict=dict, cache=dict)
@memoize(ISumNode)
def likelihood(
    node: ISumNode,
    data: np.ndarray,
    network_type: NetworkType,
    node_likelihood_dict=None,
    cache: Dict = {},
) -> np.ndarray:
    """
    Recursively calculates the likelihood for a ISumNode.
    It calls likelihood on all it children to calculate its own value by using the fitting evaluation function.

    Args:
        node:
            ISumNode to calculate likelihood for.
        data:
            Given observed or missing data.
        network_type:
            NetworkType to dispatch to this evaluation method. Expected to be SPN.
        node_likelihood_dict:
            Dictionary containing evaluation methods for SPNs.
        cache:
            Dictionary collecting which nodes were already calculated to avoid recalculating them.

    Returns: Likelihood value for ISumNode.
    """
    node_likelihood_dict = node_likelihood_dict or _node_likelihood
    inputs = [
        likelihood(child, data, SPN(), node_likelihood_dict=_node_likelihood, cache=cache)
        for child in node.children
    ]
    eval_func = node_likelihood_dict[type(node)]
    return eval_func(node, inputs)


@dispatch(IProductNode, np.ndarray, SPN, node_likelihood_dict=dict, cache=dict)
@memoize(IProductNode)
def likelihood(
    node: IProductNode,
    data: np.ndarray,
    network_type: NetworkType,
    node_likelihood_dict=None,
    cache: Dict = {},
) -> np.ndarray:
    """
    Recursively calculates the likelihood for a IProdNode.
    It calls likelihood on all it children to calculate its own value by using the fitting evaluation function.

    Args:
        node:
            IProdNode to calculate likelihood for.
        data:
            Given observed or missing data.
        network_type:
            NetworkType to dispatch to this evaluation method. Expected to be SPN.
        node_likelihood_dict:
            Dictionary containing evaluation methods for SPNs.
        cache:
            Dictionary collecting which nodes were already calculated to avoid recalculating them.

    Returns: Likelihood value for IProdNode.
    """
    node_likelihood_dict = node_likelihood_dict or _node_likelihood
    inputs = [
        likelihood(child, data, SPN(), node_likelihood_dict=_node_likelihood, cache=cache)
        for child in node.children
    ]
    eval_func = node_likelihood_dict[type(node)]
    return eval_func(node, inputs)


@dispatch(ILeafNode, np.ndarray, SPN, node_log_likelihood_dict=dict, cache=dict)
@memoize(ILeafNode)
def likelihood(
    node: ILeafNode,
    data: np.ndarray,
    network_type: NetworkType,
    node_likelihood_dict=None,
    cache: Dict = {},
) -> np.ndarray:
    """
    Calculates likelihood for a ILeafNode by evaluating the distribution represented by the leaf node type.

    Args:
        node:
            ILeafNode node to calculate likelihood for.
        data:
            Given observed or missing data.
        network_type:
            NetworkType to dispatch to this evaluation method. Expected to be SPN.
        node_likelihood_dict:
            Only used to dispatch to this function.
        cache:
            Dictionary collecting which nodes were already calculated to avoid recalculating them.

    Returns: Likelihood value for ILeafNode.
    """
    return node_likelihood(node, data=data)