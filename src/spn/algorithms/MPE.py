'''
Created on July 02, 2018

@author: Alejandro Molina
'''
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, get_nodes_by_type, eval_spn_top_down
import numpy as np


def mpe_prod(node, input_vals, data=None, lls_per_node=None, rand_gen=None):
    if len(input_vals) == 0:
        return None
    return [input_vals] * len(node.children)


def mpe_sum(node, input_vals, data=None, lls_per_node=None, rand_gen=None):
    if len(input_vals) == 0:
        return None

    w_children_log_probs = np.zeros((len(input_vals), len(node.weights)))
    for i, c in enumerate(node.children):
        w_children_log_probs[:, i] = lls_per_node[input_vals, c.id] + np.log(node.weights[i])

    max_child_branches = np.argmax(w_children_log_probs, axis=1)

    children_row_ids = []

    for i, c in enumerate(node.children):
        children_row_ids.append(input_vals[max_child_branches == i])

    return children_row_ids


def mpe_leaf(node, input_vals, data=None, lls_per_node=None, rand_gen=None):
    if len(input_vals) == 0:
        return None

    # we need to find the cells where we need to replace nans with mpes
    data_nans = np.isnan(data[input_vals, node.scope])

    n_mpe = np.sum(data_nans)

    if n_mpe == 0:
        return None

    data[input_vals[data_nans], node.scope] = node.mode


_node_mpe = {Product: mpe_prod, Sum: mpe_sum}


def add_node_mpe(node_type, lambda_func):
    _node_mpe[node_type] = lambda_func


def mpe(node, input_data, node_mpe=_node_mpe, in_place=False):
    valid, err = is_valid(node)
    assert valid, err

    assert np.all(
        np.any(np.isnan(input_data), axis=1)), "each row must have at least a nan value where the samples will be substituted"

    if in_place:
        data = input_data
    else:
        data = np.array(input_data)

    nodes = get_nodes_by_type(node)

    lls_per_node = np.zeros((data.shape[0], len(nodes)))

    # one pass bottom up evaluating the likelihoods
    log_likelihood(node, data, dtype=data.dtype, lls_matrix=lls_per_node)

    instance_ids = np.arange(data.shape[0])

    # one pass top down to decide on the max branch until it reaches a leaf, then it fills the nan slot with the mode
    eval_spn_top_down(node, node_mpe, input_vals=instance_ids, data=data, lls_per_node=lls_per_node)

    return data
