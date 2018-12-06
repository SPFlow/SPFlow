"""
Created on November 09, 2018

@author: Alejandro Molina
@author: Robert Peharz
"""

from scipy.special import logsumexp

from spn.algorithms.Gradient import gradient_backward
from spn.algorithms.Inference import log_likelihood

from spn.structure.leaves.parametric.Parametric import Gaussian

from spn.structure.Base import Sum, get_nodes_by_type, get_number_of_nodes
import numpy as np


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
