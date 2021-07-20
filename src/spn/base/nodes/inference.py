"""
Created on July 08, 2021

@authors: Kevin Huy Nguyen

This file provides the inference methods for SPNs.
"""

import numpy as np
from numpy import ndarray
from scipy.special import logsumexp  # type: ignore
from spn.base.nodes.node import LeafNode, ProductNode, SumNode, eval_spn_bottom_up, Node
from spn.base.nodes.leaves.parametric.inference import node_likelihood, node_log_likelihood
from spn.base.nodes.leaves.parametric.parametric import Gaussian
from typing import List, Callable, Type, Optional, Dict

from multipledispatch import dispatch  # type: ignore


def prod_log_likelihood(node: ProductNode, children: List[ndarray], **kwargs):
    """
    Calculates the log-likelihood for a product node.

    Args:
        node:
            ProductNode to calculate log-likelihood for.
        children:
            np.array of child node values of ProductNode.

    Returns: Log-likelihood value for product node.
    """

    llchildren: ndarray = np.concatenate(children, axis=1)
    pll: ndarray = np.sum(llchildren, axis=1).reshape(-1, 1)
    pll[np.isinf(pll)] = np.finfo(pll.dtype).min

    return pll


def prod_likelihood(node: ProductNode, children: List[ndarray], **kwargs):
    """
    Calculates the likelihood for a product node.

    Args:
        node:
            ProductNode to calculate likelihood for.
        children:
            np.array of child node values of ProductNode.

    Returns: likelihood value for product node.
    """

    llchildren: ndarray = np.concatenate(children, axis=1)

    return np.prod(llchildren, axis=1).reshape(-1, 1)


def sum_log_likelihood(node: SumNode, children: List[ndarray], **kwargs):
    """
    Calculates the log-likelihood for a sum node.

    Args:
        node:
            SumNode to calculate log-likelihood for.
        children:
            np.array of child node values of SumNode.

    Returns: Log-likelihood value for sum node.
    """

    llchildren: ndarray = np.concatenate(children, axis=1)
    b: ndarray = node.weights
    sll: ndarray = logsumexp(llchildren, b=b, axis=1).reshape(-1, 1)

    return sll


def sum_likelihood(node: SumNode, children: List[ndarray], **kwargs):
    """
    Calculates the likelihood for a sum node.

    Args:
        node:
            SumNode to calculate likelihood for.
        children:
            np.array of child node values of SumNode.

    Returns: Likelihood value for sum node.
    """

    llchildren: ndarray = np.concatenate(children, axis=1)
    b: ndarray = np.array(node.weights)

    return np.dot(llchildren, b).reshape(-1, 1)


_node_log_likelihood: Dict[Type, Callable] = {
    SumNode: sum_log_likelihood,
    ProductNode: prod_log_likelihood,
    LeafNode: node_log_likelihood,
}
_node_likelihood: Dict[Type, Callable] = {
    SumNode: sum_likelihood,
    ProductNode: prod_likelihood,
    LeafNode: node_likelihood,
}

# TODO: **kwargs not supported
@dispatch(Node, ndarray, node_likelihood=dict)
def likelihood(node: Node, data: ndarray, node_likelihood: Dict[Type, Callable] = _node_likelihood) -> ndarray:
    """
    Calculates the likelihood for a SPN.

    Args:
        node:
            Root node of SPN to calculate likelihood for.
        data:
            Data given to evaluate LeafNodes.
        node_likelihood:
            dictionary that contains k: Class of the node, v: lambda function that receives as parameters (node, args**)
            for leaf nodes and (node, [children results], args**) for other nodes.

    Returns: Likelihood value for SPN.
    """

    all_results: Optional[Dict[Node, ndarray]] = {}
    result: ndarray = eval_spn_bottom_up(node, node_likelihood, all_results=all_results, data=data)
    return result


# TODO: **kwargs not supported
@dispatch(Node, ndarray, node_log_likelihood=dict)
def log_likelihood(
    node: Node, data: ndarray, node_log_likelihood: Dict[Type, Callable] = _node_log_likelihood
) -> ndarray:
    """
    Calculates the log-likelihood for a SPN.

    Args:
        node:
            Root node of SPN to calculate log-likelihood for.
        data:
            Data given to evaluate LeafNodes.
        node_log_likelihood:
            dictionary that contains k: Class of the node, v: lambda function that receives as parameters (node, args**)
            for leaf nodes and (node, [children results], args**) for other nodes.

    Returns: Log-likelihood value for SPN.
    """
    return likelihood(node, data, node_likelihood=node_log_likelihood)


if __name__ == "__main__":
    spn = SumNode(
        children=[
            ProductNode(
                children=[
                    Gaussian(scope=[0], mean=0, stdev=1.0),
                    SumNode(
                        children=[
                            ProductNode(
                                children=[
                                    Gaussian(scope=[1], mean=0, stdev=1.0),
                                    Gaussian(scope=[2], mean=0, stdev=1.0),
                                ],
                                scope=[1, 2],
                            ),
                            ProductNode(
                                children=[
                                    Gaussian(scope=[1], mean=0, stdev=1.0),
                                    Gaussian(scope=[2], mean=0, stdev=1.0),
                                ],
                                scope=[1, 2],
                            ),
                        ],
                        scope=[1, 2],
                        weights=np.array([0.3, 0.7]),
                    ),
                ],
                scope=[0, 1, 2],
            ),
            ProductNode(
                children=[
                    Gaussian(scope=[0], mean=0, stdev=1.0),
                    Gaussian(scope=[1], mean=0, stdev=1.0),
                    Gaussian(scope=[2], mean=0, stdev=1.0),
                ],
                scope=[0, 1, 2],
            ),
        ],
        scope=[0, 1, 2],
        weights=np.array([0.4, 0.6]),
    )

    result = likelihood(spn, np.array([1.0, 0.0, 1.0]).reshape(-1, 3))
    print(result, np.log(result))

    result = log_likelihood(spn, np.array([1.0, 0.0, 1.0]).reshape(-1, 3))
    print(np.exp(result), result)

    result = likelihood(spn, np.array([np.nan, 0.0, 1.0]).reshape(-1, 3))
    print(result, np.log(result))

    result = log_likelihood(spn, np.array([np.nan, 0.0, 1.0]).reshape(-1, 3))
    print(np.exp(result), result)

    # [[0.023358]] [[-3.7568156]]
    # [[0.09653235]] [[-2.33787707]] marginallize rv with scope 0
