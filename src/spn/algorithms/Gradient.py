import numpy as np

from spn.algorithms.Inference import likelihood
from spn.algorithms.Condition import condition
from spn.structure.Base import Sum, Product, eval_spn_bottom_up


_node_gradients = {}


def add_node_gradients(node_type, lambda_func):
    _node_gradients[node_type] = lambda_func


def sum_gradient_forward(node, children, input_vals, dtype=np.float64):
    b = np.array(node.weights, dtype=dtype)
    gradient_children = np.array([weight * child for weight, child in zip(b, children)])
    gradient_children = np.sum(gradient_children, axis=0)
    return gradient_children


def prod_gradient_forward(node, children, input_vals, dtype=np.float64):
    shape = input_vals.shape

    probs = np.concatenate([likelihood(c, input_vals) for c in node.children], axis=1)

    joined = np.zeros(shape)
    joined[:] = np.nan
    for i, c in enumerate(children):
        joined[:, node.children[i].scope] = c

    results = np.zeros(shape)
    results[:] = np.nan
    for index in range(shape[1]):
        mask = np.full(shape, False)
        mask[:, index] = True
        array = np.ma.masked_array(probs, mask=mask)
        results[:, index] = np.prod(array, axis=1) * joined[:, index]
    return results


def gradient(spn, evidence):
    """
    Computes a forward propagated gradient through the spn. This function
    currently assumes a tree structured SPN!

    :param spn:
    :param evidence:
    :return:
    """
    _node_gradients[Sum] = sum_gradient_forward
    _node_gradients[Product] = prod_gradient_forward

    node_gradients = _node_gradients

    gradients = eval_spn_bottom_up(spn, node_gradients, input_vals=evidence)
    return gradients


def conditional_gradient(spn, conditional_evidence, gradient_evidence):
    cond_spn = condition(spn, conditional_evidence)
    keep_idx = np.isnan(conditional_evidence)[0]
    evidence = gradient_evidence[:, keep_idx]
    return gradient(cond_spn, evidence)