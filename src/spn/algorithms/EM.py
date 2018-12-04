"""
Created on November 09, 2018

@author: Alejandro Molina
@author: Robert Peharz
"""

from scipy.special import logsumexp

from spn.algorithms.Inference import log_likelihood

from spn.structure.leaves.parametric.Parametric import Gaussian

from spn.structure.Base import eval_spn_top_down, Sum, Product, get_nodes_by_type, get_number_of_nodes, Leaf
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


def gaussian_em_update(
    node, node_lls=None, node_gradients=None, root_lls=None, data=None, update_mean=True, update_std=True, **kwargs
):
    p = (node_gradients - root_lls) + node_lls
    lse = logsumexp(p)
    w = np.exp(p - lse)
    X = data[:, node.scope[0]]

    mean = np.sum(w * X)

    if update_mean:
        node.mean = mean

    if update_std:
        dev = np.power(X - mean, 2)
        node.std = np.sqrt(np.sum(w * dev))


def sum_em_update(node, node_gradients=None, root_lls=None, all_lls=None, **kwargs):
    RinvGrad = node_gradients - root_lls
    for i, c in enumerate(node.children):
        new_w = RinvGrad + all_lls[:, c.id] + np.log(node.weights[i])
        node.weights[i] = logsumexp(new_w)
    total_weight = np.sum(node.weights)
    node.weights = (node.weights / total_weight).tolist()


_node_updates = {Gaussian: gaussian_em_update, Sum: sum_em_update}


def EM_optimization(spn, data, iterations=5, node_updates=_node_updates, **kwargs):
    for _ in range(iterations):
        lls_per_node = np.zeros((data.shape[0], get_number_of_nodes(spn)))

        # one pass bottom up evaluating the likelihoods
        log_likelihood(spn, data, dtype=data.dtype, lls_matrix=lls_per_node)

        gradients = gradient_backward(spn, lls_per_node)

        R = lls_per_node[:, 0]

        for node_type, func in node_updates.items():  # TODO: do in parallel
            for node in get_nodes_by_type(spn, node_type):
                func(
                    node,
                    node_lls=lls_per_node[:, node.id],
                    node_gradients=gradients[:, node.id],
                    root_lls=R,
                    all_lls=lls_per_node,
                    all_gradients=gradients,
                    data=data,
                    **kwargs
                )
