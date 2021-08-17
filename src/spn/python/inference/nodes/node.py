"""
Created on July 08, 2021

@authors: Kevin Huy Nguyen

This file provides the inference methods for SPNs.
"""

import numpy as np
from numpy import ndarray
from scipy.special import logsumexp  # type: ignore
from spn.python.structure.nodes.node import Node, SumNode, ProductNode, LeafNode, get_topological_order
#from spn.python.structure.
from spn.python.structure.nodes.structural_marginalization import marginalize
from spn.python.structure.nodes import Gaussian
from spn.python.inference.nodes.leaves.parametric import node_likelihood, node_log_likelihood
from typing import List, Callable, Type, Optional, Dict

from multipledispatch import dispatch  # type: ignore

def eval_spn_bottom_up(
    node: Node,
    eval_functions: Dict[Type, Callable],
    all_results: Optional[Dict[Node, np.ndarray]] = None,
    **args,
) -> np.ndarray:
    """
    Evaluates the spn bottom up using functions specified for node types.

    Args:
        node:
            SPN root node.
        eval_functions:
            dictionary that contains k: Class of the node, v: lambda function that receives as parameters (node, args**)
            for leaf nodes and (node, [children results], args**) for other nodes.
        all_results: dictionary that contains k: node, v: result of evaluation of the lambda
                        function for that node. Used to store intermediate results so non-tree graphs can be
                        computed in O(n) size of the network.
        args: free parameters that will be fed to the lambda functions.

    Returns: Result of computing and propagating all the values through the network.
    """

    nodes: List[Node] = get_topological_order(node)

    if all_results is None:
        all_results = {}
    else:
        all_results.clear()
    node_type_eval_func_dict: Dict[Type, List[Callable]] = {}
    node_type_is_leaf_dict: Dict[Type, bool] = {}
    for node_type, func in eval_functions.items():
        if node_type not in node_type_eval_func_dict:
            node_type_eval_func_dict[node_type] = []
        node_type_eval_func_dict[node_type].append(func)
        node_type_is_leaf_dict[node_type] = issubclass(node_type, LeafNode)
    leaf_func: Optional[Callable] = eval_functions.get(LeafNode, None)

    tmp_children_list: List[Optional[np.ndarray]] = []
    len_tmp_children_list: int = 0
    for n in nodes:
        try:
            func = node_type_eval_func_dict[type(n)][-1]
            n_is_leaf: bool = node_type_is_leaf_dict[type(n)]
        except:
            if isinstance(n, LeafNode) and leaf_func is not None:
                func = leaf_func
                n_is_leaf = True
            else:
                raise AssertionError(
                    "No lambda function associated with type: %s" % type(n).__name__
                )

        if n_is_leaf:
            result: np.ndarray = func(n, **args)
        else:
            len_children: int = len(n.children)
            if len_tmp_children_list < len_children:
                tmp_children_list.extend([None] * len_children)
                len_tmp_children_list = len(tmp_children_list)
            for i in range(len_children):
                ci: Node = n.children[i]
                tmp_children_list[i] = all_results[ci]
            result = func(n, tmp_children_list[0:len_children], **args)
        all_results[n] = result

    for node_type, func in eval_functions.items():
        del node_type_eval_func_dict[node_type][-1]
        if len(node_type_eval_func_dict[node_type]) == 0:
            del node_type_eval_func_dict[node_type]
    return all_results[node]


def prod_log_likelihood(node: ProductNode, children: List[ndarray], **kwargs) -> ndarray:
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


def prod_likelihood(node: ProductNode, children: List[ndarray], **kwargs) -> ndarray:
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


def sum_log_likelihood(node: SumNode, children: List[ndarray], **kwargs) -> ndarray:
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


def sum_likelihood(node: SumNode, children: List[ndarray], **kwargs) -> ndarray:
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


@dispatch(Node, ndarray, node_likelihood=dict)
def likelihood(
    node: Node, data: ndarray, node_likelihood: Dict[Type, Callable] = _node_likelihood
) -> ndarray:
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
    spn_marg = marginalize(spn, [1, 2])

    result = likelihood(spn, np.array([1.0, 0.0, 1.0]).reshape(-1, 3))
    print(result, np.log(result))

    result = log_likelihood(spn, np.array([1.0, 0.0, 1.0]).reshape(-1, 3))
    print(np.exp(result), result)

    result = likelihood(spn, np.array([np.nan, 0.0, 1.0]).reshape(-1, 3))
    print(result, np.log(result))

    result = log_likelihood(spn, np.array([np.nan, 0.0, 1.0]).reshape(-1, 3))
    print(np.exp(result), result)

    l_marg = likelihood(spn_marg, np.array([1.0, 0.0, 1.0]).reshape(-1, 3))
    print(l_marg, np.log(l_marg))

    ll_marg = log_likelihood(spn_marg, np.array([1.0, 0.0, 1.0]).reshape(-1, 3))
    print(np.exp(ll_marg), ll_marg)

    # [[0.023358]] [[-3.7568156]]
    # [[0.09653235]] [[-2.33787707]] marginallize rv with scope 0
