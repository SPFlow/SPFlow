'''
Created on November 09, 2018

@author: Alejandro Molina
@author: Robert Peharz
'''

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

    eval_spn_top_down(spn, node_gradients, parent_result=np.zeros((lls_per_node.shape[0])), gradient_result=gradient_result,
                      lls_per_node=lls_per_node)

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


def gaussian_em_update(node, lls, gradients, root_lls, data):
    p = (gradients - root_lls) + lls
    w = np.exp(p)
    w = w / np.sum(w)
    node.mean = np.sum(w * node.mean)
    print(node.mean)


_leaf_node_updates = {Gaussian: gaussian_em_update}


def EM_optimization(spn, data, iterations=5, leaf_node_updates=_leaf_node_updates):
    for _ in range(iterations):
        lls_per_node = np.zeros((data.shape[0], get_number_of_nodes(spn)))

        # one pass bottom up evaluating the likelihoods
        log_likelihood(spn, data, dtype=data.dtype, lls_matrix=lls_per_node)

        gradients = gradient_backward(spn, lls_per_node)

        R = lls_per_node[:, 0]

        for sum_node in get_nodes_by_type(spn, Sum):
            RinvGrad = (gradients[:, sum_node.id] - R)
            for i, c in enumerate(sum_node.children):
                new_w = RinvGrad + lls_per_node[:, c.id] + np.log(sum_node.weights[i])
                sum_node.weights[i] = logsumexp(new_w)
            total_weight = np.sum(sum_node.weights)
            sum_node.weights = (sum_node.weights / total_weight).tolist()

        for leaf_node in get_nodes_by_type(spn, Leaf):
            f = leaf_node_updates[leaf_node.__class__]
            f(leaf_node, lls_per_node[:, leaf_node.id], gradients[:, leaf_node.id], R, data)
