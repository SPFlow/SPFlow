import numpy as np

from spn.algorithms.Condition import condition
from spn.algorithms.Inference import _node_likelihood
from spn.algorithms.Marginalization import marginalize
from spn.structure.Base import Sum, Product, eval_spn_bottom_up, Leaf, get_node_types

_node_moment = {}


def add_node_moment(node_type, lambda_func):
    _node_moment[node_type] = lambda_func


def prod_moment(node, children, order=1, result_array=None, dtype=np.float64):
    joined = np.zeros((1, len(node.scope)))
    joined[:] = np.nan
    for i, c in enumerate(children):
        joined[:, node.children[i].scope] = c[:, node.children[i].scope]
    return joined


def sum_moment(node, children, order=1, result_array=None, dtype=np.float64):
    joined_children = np.array(children)[:, 0, :]
    b = np.array(node.weights, dtype=dtype)
    weighted = np.sum((joined_children * b[:, np.newaxis]), 0, keepdims=True)
    return weighted


def leaf_moment(_node_moment, _node_likelihood):
    def leaf_moment_function(node, order=1, result_array=None):
        leaf_data = result_array[:, node.scope]
        moment_indices = np.isnan(leaf_data)
        shape = result_array.shape
        result = result_array[:, node.scope]
        moment = _node_moment(node, order)
        data = np.full(result.shape, np.nan)
        data[moment_indices] = moment
        full_data = np.full(shape, np.nan)
        full_data[:, node.scope] = data
        return full_data

    return leaf_moment_function


def ConditionalMoment(
    spn, evidence, feature_scope, node_moment=_node_moment, node_likelihoods=_node_likelihood, order=1
):
    """
    Computes a conditional moment given a numpy array of evidence
    :param spn: a valid spn
    :param feature_scope: optional list of features on which to compute the moments
    :param evidence: the evidence for the conditioning step
    :param node_moment: optional list of node moment functions
    :param node_likelihoods: optional list of node likelihood functions
    :param order: the order of the moment to compute
    :return: an np array of computed moments
    """
    assert feature_scope is not None, "When using evidence a feature scope needs to be passed"

    feature_scope = list(feature_scope)
    assert np.all(np.isnan(evidence[:, feature_scope])), "Evidence cannot be requested for features in scope"

    all_results = []
    for line in evidence:
        cond_spn = condition(spn, line.reshape(1, -1))
        moment = Moment(cond_spn, feature_scope, node_moment, node_likelihoods, order=order)
        all_results.append(moment)
    if feature_scope:
        output_size = (evidence.shape[0], len(feature_scope))
    else:
        output_size = evidence.shape
    return np.array(all_results).reshape(output_size)


def Moment(spn, feature_scope=None, node_moment=_node_moment, node_likelihoods=_node_likelihood, order=1):
    """
    Computes moments from an spn
    :param spn: a valid spn
    :param feature_scope: optional list of features on which to compute the moments
    :param node_moment: optional list of node moment functions
    :param node_likelihoods: optional list of node likelihood functions
    :param order: the order of the moment to compute
    :return: an np array of computed moments
    """

    if feature_scope is None:
        feature_scope = spn.scope
    feature_scope = list(feature_scope)

    assert len(feature_scope) == len(list(feature_scope)), "Found double entries in feature list"

    marg_spn = marginalize(spn, feature_scope)

    node_moments = {Sum: sum_moment, Product: prod_moment}

    for node in get_node_types(marg_spn, Leaf):
        try:
            moment = node_moment[node]
            node_ll = node_likelihoods[node]
        except KeyError:
            raise AssertionError("Node type {} doe not have associated moment and likelihoods".format(node))
        node_moments[node] = leaf_moment(moment, node_ll)

    results = np.full((1, max(spn.scope) + 1), np.nan)

    moment = eval_spn_bottom_up(marg_spn, node_moments, order=order, result_array=results)
    return moment[:, feature_scope]


def get_mean(spn):
    """
    Small utility function to complete the full list of first order moments
    (means) from a given SPN
    :param spn: the spn
    :return:
    """
    return Moment(spn)


def get_variance(spn):
    """
    Small utility function to complete the full list of second order
    centralized moments (variances) from a given SPN
    :param spn: the spn
    :return:
    """

    return Moment(spn, order=2) - get_mean(spn) ** 2
