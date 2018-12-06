from spn.structure.Base import eval_spn_top_down, Sum, Product, Leaf
import numpy as np


def gradient_backward(spn, lls_per_node):
    node_gradients = {}
    node_gradients[Sum] = sum_gradient_backward
    node_gradients[Product] = prod_gradient_backward
    node_gradients[Leaf] = leaf_gradient_backward

    gradient_result = np.zeros_like(lls_per_node)

    eval_spn_top_down(
        spn,
        node_gradients,
        parent_result=np.zeros((lls_per_node.shape[0])),
        gradient_result=gradient_result,
        lls_per_node=lls_per_node,
    )

    return gradient_result


def leaf_gradient_backward(node, parent_result, gradient_result=None, lls_per_node=None):
    gradients = np.zeros((parent_result.shape[0]))
    gradients[:] = parent_result  # log_sum_exp

    gradient_result[:, node.id] = gradients


def sum_gradient_backward(node, parent_result, gradient_result=None, lls_per_node=None):
    gradients = np.zeros((parent_result.shape[0]))
    gradients[:] = parent_result  # log_sum_exp

    gradient_result[:, node.id] = gradients

    messages_to_children = []

    for i, c in enumerate(node.children):
        messages_to_children.append(gradients + np.log(node.weights[i]))

    assert not np.any(np.isnan(messages_to_children)), "Nans found in iteration"

    return messages_to_children


def prod_gradient_backward(node, parent_result, gradient_result=None, lls_per_node=None):
    gradients = np.zeros((parent_result.shape[0]))
    gradients[:] = parent_result  # log_sum_exp

    gradient_result[:, node.id] = gradients

    messages_to_children = []

    # TODO handle zeros for efficiency, darwiche 2003
    output_ll = lls_per_node[:, node.id]

    for i, c in enumerate(node.children):
        messages_to_children.append(output_ll - lls_per_node[:, c.id])

    assert not np.any(np.isnan(messages_to_children)), "Nans found in iteration"

    return messages_to_children
