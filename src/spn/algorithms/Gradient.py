import numpy as np
from scipy.special import logsumexp

from spn.algorithms.Inference import log_likelihood
from spn.structure.Base import eval_spn_top_down, Sum, Product, Leaf, get_number_of_nodes, get_nodes_by_type


def merge_gradients(parent_gradients):
    return logsumexp(np.concatenate(parent_gradients).reshape(-1, 1), axis=1)


def leaf_gradient_backward(node, parent_result, gradient_result=None, lls_per_node=None):
    parent_gradients = merge_gradients(parent_result)

    gradients = np.zeros((parent_gradients.shape[0]))
    gradients[:] = parent_gradients

    gradient_result[:, node.id] = gradients


def sum_gradient_backward(node, parent_result, gradient_result=None, lls_per_node=None):
    parent_gradients = merge_gradients(parent_result)

    gradients = np.zeros((parent_gradients.shape[0]))
    gradients[:] = parent_gradients

    gradient_result[:, node.id] = gradients

    messages_to_children = {}
    wlog = np.log(node.weights)

    for i, c in enumerate(node.children):
        children_gradient = gradients + wlog[i]
        children_gradient[np.isinf(children_gradient)] = np.finfo(gradient_result.dtype).min
        messages_to_children[c] = children_gradient

        assert not np.any(np.isnan(children_gradient)), "Nans found in iteration"
        assert not np.any(np.isinf(children_gradient)), "inf found in iteration"

    return messages_to_children


def prod_gradient_backward(node, parent_result, gradient_result=None, lls_per_node=None):
    parent_gradients = merge_gradients(parent_result)

    gradients = np.zeros((parent_gradients.shape[0]))
    gradients[:] = parent_gradients

    gradient_result[:, node.id] = gradients

    messages_to_children = {}

    output_ll = lls_per_node[:, node.id]

    for i, c in enumerate(node.children):
        children_gradient = gradients + output_ll - lls_per_node[:, c.id]
        children_gradient[np.isinf(children_gradient)] = np.finfo(gradient_result.dtype).min
        messages_to_children[c] = children_gradient

        assert not np.any(np.isnan(children_gradient)), "Nans found in iteration"
        assert not np.any(np.isinf(children_gradient)), "inf found in iteration"

    return messages_to_children


_node_gradients = {Sum: sum_gradient_backward, Product: prod_gradient_backward, Leaf: leaf_gradient_backward}
_node_feature_gradients = {}


def add_node_gradient(node_type, lambda_func):
    _node_gradients[node_type] = lambda_func


def add_node_feature_gradient(node_type, lambda_func):
    _node_feature_gradients[node_type] = lambda_func


def gradient_backward(spn, lls_per_node, node_gradients=_node_gradients):
    gradient_result = np.zeros_like(lls_per_node)

    eval_spn_top_down(
        spn,
        node_gradients,
        parent_result=np.zeros((lls_per_node.shape[0])),
        gradient_result=gradient_result,
        lls_per_node=lls_per_node,
    )

    return gradient_result


def feature_gradient(node, data, node_gradient_functions=_node_feature_gradients, lls_per_node=None):
    """
    Feature gradients are computed for the input query and each feature using
    the backwards automatic differentiation. In mathematicl terms, it computes the
    partial derivatives \partial P(X) / \partial X_i
 

    :param node: Node for the gradient calculation
    :param data: data for the computation. NaN values are implicitely marginalized out
    :param lls_per_node: optional for storing the intermediate results
    """

    all_leaves = get_nodes_by_type(node, Leaf)

    if not lls_per_node:
        lls_per_node = np.full((data.shape[0], get_number_of_nodes(node)), np.nan)
    log_likelihood(node, data, lls_matrix=lls_per_node)

    gradients = np.exp(gradient_backward(node, lls_per_node))

    node_gradients = []

    for spn_node in all_leaves:
        i = spn_node.id
        result = node_gradient_functions[type(spn_node)](spn_node, data)
        node_gradients.append(result * gradients[:, i].reshape(-1, 1))

    node_gradients = np.array(node_gradients)

    return np.nansum(node_gradients, axis=0)
