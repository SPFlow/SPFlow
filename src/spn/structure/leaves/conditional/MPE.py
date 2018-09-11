'''
Created on July 02, 2018

@author: Alejandro Molina
'''
from spn.algorithms import add_node_mpe
from spn.structure.leaves.conditional.Conditional import Conditional_Gaussian, Conditional_Poisson, \
    Conditional_Bernoulli
import numpy as np

from spn.structure.leaves.conditional.utils import get_scipy_obj_params


def get_conditional_params(node, input_vals, data):
    if len(input_vals) == 0:
        return None, None

    # we need to find the cells where we need to replace nans with mpes
    data_nans = np.isnan(data[input_vals, node.scope])

    n_mpe = np.sum(data_nans)

    if n_mpe == 0:
        return None, None

    _, params = get_scipy_obj_params(node, data[:, -node.evidence_size:])
    return data_nans, params


def conditional_gaussian_mpe_leaf(node, input_vals, data=None, lls_per_node=None, rand_gen=None):
    data_nans, params = get_conditional_params(node, input_vals, data)
    if data_nans is None:
        return

    data[input_vals[data_nans], node.scope] = params["loc"]


def conditional_poisson_mpe_leaf(node, input_vals, data=None, lls_per_node=None, rand_gen=None):
    data_nans, params = get_conditional_params(node, input_vals, data)
    if data_nans is None:
        return

    data[input_vals[data_nans], node.scope] = np.floor(params["mu"])


def conditional_bernoulli_mpe_leaf(node, input_vals, data=None, lls_per_node=None, rand_gen=None):
    data_nans, params = get_conditional_params(node, input_vals, data)

    if data_nans is None:
        return
    p = params["p"]
    q = 1 - p
    data[input_vals[data_nans], node.scope] = (q <= p).astype(int)


def add_conditional_mpe_support():
    add_node_mpe(Conditional_Gaussian, conditional_gaussian_mpe_leaf)
    add_node_mpe(Conditional_Poisson, conditional_poisson_mpe_leaf)
    add_node_mpe(Conditional_Bernoulli, conditional_bernoulli_mpe_leaf)
