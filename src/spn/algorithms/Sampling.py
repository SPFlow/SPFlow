'''
Created on April 5, 2018

@author: Alejandro Molina
'''
import logging

import numpy as np

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, get_nodes_by_type, eval_spn_top_down


def sample_prod(node, input_vals, data=None, lls_per_node=None, rand_gen=None):
    if len(input_vals) == 0:
        return None
    return [input_vals] * len(node.children)


def sample_sum(node, input_vals, data=None, lls_per_node=None, rand_gen=None):
    if len(input_vals) == 0:
        return None

    w_children_log_probs = np.zeros((len(input_vals), len(node.weights)))
    for i, c in enumerate(node.children):
        w_children_log_probs[:, i] = lls_per_node[input_vals, c.id] + np.log(node.weights[i])

    z_gumbels = rand_gen.gumbel(loc=0, scale=1,
                                size=(w_children_log_probs.shape[0], w_children_log_probs.shape[1]))
    g_children_log_probs = w_children_log_probs + z_gumbels
    rand_child_branches = np.argmax(g_children_log_probs, axis=1)

    children_row_ids = []

    for i, c in enumerate(node.children):
        children_row_ids.append(input_vals[rand_child_branches == i])

    return children_row_ids


def sample_leaf(node, input_vals, data=None, lls_per_node=None, rand_gen=None):
    if len(input_vals) == 0:
        return None

    # we need to find the cells where we need to replace nans with samples
    data_nans = np.isnan(data[input_vals, node.scope])

    n_samples = np.sum(data_nans)

    if n_samples == 0:
        return None

    data[input_vals[data_nans], node.scope] = _leaf_sampling[type(node)](node, n_samples=n_samples,
                                                                         data=data[input_vals[data_nans], :],
                                                                         rand_gen=rand_gen)


_node_sampling = {Product: sample_prod, Sum: sample_sum}
_leaf_sampling = {}


def add_node_sampling(node_type, lambda_func):
    _leaf_sampling[node_type] = lambda_func
    _node_sampling[node_type] = sample_leaf


def sample_instances(node, input_data, rand_gen, node_sampling=_node_sampling, in_place=False):
    """
    Implementing hierarchical sampling

    """

    # first, we do a bottom-up pass to compute the likelihood taking into account marginals.
    # then we do a top-down pass, to sample taking into account the likelihoods.

    if in_place:
        data = input_data
    else:
        data = np.array(input_data)

    valid, err = is_valid(node)
    assert valid, err

    assert np.all(
        np.any(np.isnan(data), axis=1)), "each row must have at least a nan value where the samples will be substituted"

    nodes = get_nodes_by_type(node)

    lls_per_node = np.zeros((data.shape[0], len(nodes)))

    log_likelihood(node, data, dtype=data.dtype, lls_matrix=lls_per_node)

    instance_ids = np.arange(data.shape[0])

    eval_spn_top_down(node, node_sampling, parent_result=instance_ids, data=data, lls_per_node=lls_per_node,
                      rand_gen=rand_gen)

    return data
