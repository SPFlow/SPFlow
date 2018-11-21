import numpy as np

from spn.algorithms.Inference import likelihood
from spn.algorithms.Condition import condition
from spn.structure.Base import Sum, Product, eval_spn_bottom_up, eval_spn_top_down


_node_gradients = {}


def add_node_gradients(node_type, lambda_func):
    _node_gradients[node_type] = lambda_func


def sum_gradient_forward(node, children, input_vals, dtype=np.float64):
    b = np.array(node.weights, dtype=dtype)
    joined_c = np.stack(children)
    gradient_children = np.array([weight * tensor for weight, tensor in zip(b, joined_c)])
    results = np.sum(gradient_children, axis=0)
    assert len(node.scope) == results.shape[1], '{} vs {}'.format(node.scope, results.shape)
    return results


def prod_gradient_forward(node, children, input_vals, dtype=np.float64):
    probs = [likelihood(c, input_vals) for c in node.children]

    results = []

    for index, c in enumerate(children):
        array = np.prod([l for i, l in enumerate(probs) if i != index], axis=0)
        results.append(array * c)

    results = np.concatenate(results, axis=1)

    assert len(node.scope) == results.shape[1], '{} vs {}: got {}'.format(node.scope, results.shape, [c.shape for c in children])
    return results


def gradient_forward(spn, evidence):
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


def sum_gradient_backward():
    pass


def prod_gradient_backward():
    pass


def backprop_gradient(spn, evidence):
    probs = likelihood(spn, evidence)
    _node_gradients[Sum] = sum_gradient_backward
    _node_gradients[Product] = prod_gradient_backward

    node_gradients = _node_gradients

    all_gradients = {}
    gradients_input = eval_spn_top_down(
        spn, node_gradients, all_results=all_gradients, probs=probs)

    return gradients_input


def conditional_gradient(spn, conditional_evidence, gradient_evidence):
    print(conditional_evidence)
    cond_spn = condition(spn, conditional_evidence)
    gradients = gradient_forward(cond_spn, gradient_evidence)
    return gradients
