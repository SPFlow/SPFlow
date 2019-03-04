"""
Created on April 5, 2018

@author: Alejandro Molina
"""
import logging

import numpy as np

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Leaf, Product, Sum, get_nodes_by_type, eval_spn_top_down
from spn.structure.leaves.parametric.Parametric import Bernoulli
from scipy.misc import logsumexp
import logging

logger = logging.getLogger(__name__)

def marginal_prob_prod(node, parent_result, data=None, lls_per_node=None, leaf_node_prob=None):
    # Sanity check
    assert len(parent_result) == 1, "{func}: parent_result should be of length 1, got {length}".format(func=__name__, length=len(parent_result))
    return [parent_result] * len(node.children)

def marginal_prob_sum(node, parent_result, data=None, lls_per_node=None, leaf_node_prob=None):

    num_child = len(node.children)

    # w_children_log_prob is the probability of selecting a child on top-down pass. 
    w_children_log_probs = np.zeros( (lls_per_node.shape[0], num_child ) )
    for i, c in enumerate(node.children):
        w_children_log_probs[:, i] = lls_per_node[:, c.id] + np.log(node.weights[i])

    # Normalizing. Is this necessary? 
    w_children_logsumexp = logsumexp(w_children_log_probs, axis=1)
    w_children_logsumexp = np.tile( logsumexp(w_children_log_probs, axis=1), (1, num_child) )

    w_parent_log_prob = np.tile( parent_result, (1, num_child) )

    # Total probability of selecting a children is self times parent (last term is for normalizing)    
    log_prob_children_selection = w_parent_log_prob + w_children_log_probs - w_children_logsumexp
    log_prob_children_selection = log_prob_children_selection.reshape(-1, 1)

    assert len(log_prob_children_selection) == num_child, "length is {length}, num_child is {num_child}".format(
        length=w_parent_log_prob, num_child=num_child
        )

    return log_prob_children_selection

def marginal_prob_leaf(node, parent_result, data=None, lls_per_node=None, leaf_node_prob=None):

    # Get the mixture probability at leaf_node. This allocates more space than needed atm. 
    leaf_node_prob[:, node.id] = np.exp(parent_result)


_node_marginal_prob = {Product: marginal_prob_prod, Sum: marginal_prob_sum, Bernoulli: marginal_prob_leaf}

def add_node_marginal_prob(node_type, lambda_func):
    _node_marginal_prob[node_type] = lambda_func

def get_marginal_prob(node, input_data, marginal_prob_funcs=_node_marginal_prob):
    """
    Get marginal probability at unseen input variables given input_data (i.e. the probability of a node being selected)
    """

    data = np.array(input_data)

    valid, err = is_valid(node)
    assert valid, err

    # first, we do a bottom-up pass to compute the likelihood taking into account marginals.
    nodes = get_nodes_by_type(node)
    lls_per_node = np.zeros((data.shape[0], len(nodes)))
    log_likelihood(node, data, dtype=data.dtype, lls_matrix=lls_per_node)

    # then we do a top-down pass, to sample taking into account the likelihoods.
    # Keep track of probability of individual leaf node selection. 
    leaf_node_prob = np.zeros(shape=(data.shape[0], len(nodes)), dtype='float32')
    eval_spn_top_down(
        node, 
        marginal_prob_funcs, 
        parent_result=np.zeros( shape=(data.shape[0], 1) ), 
        data=data, 
        lls_per_node=lls_per_node, 
        leaf_node_prob=leaf_node_prob
    )

    return leaf_node_prob

def get_marginal_prob_bernoulli(node, input_data, marginal_prob_funcs=_node_marginal_prob): 

    data = np.array(input_data)

    leaf_node_prob = get_marginal_prob(node, data, marginal_prob_funcs=marginal_prob_funcs)
    leaf_nodes = get_nodes_by_type(node, ntype=Bernoulli)
 
    marginal_prob = np.zeros(shape=data.shape, dtype='float32')

    # Probability of drawing true at a input node: probability of selecting a mixture component * probability of that mixture component returning true 
    for leaf_node in leaf_nodes:
        marginal_prob[:, leaf_node.scope[0]] += leaf_node_prob[:, leaf_node.id] * leaf_node.p

    return marginal_prob
