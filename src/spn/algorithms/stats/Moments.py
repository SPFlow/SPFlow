import numpy as np

from spn.algorithms.Condition import condition
from spn.algorithms.Inference import likelihood, _node_likelihood
from spn.algorithms.Marginalization import marginalize
from spn.structure.Base import Sum, Product, eval_spn_bottom_up, get_leaf_types

_node_moment = {}


def add_node_moment(node_type, lambda_func):
    _node_moment[node_type] = lambda_func


def prod_moment(node, children, order=1, evidence=None, dtype=np.float64):
    joined = np.zeros((1, len(node.scope)))
    joined[:] = np.nan
    for i, c in enumerate(children):
        joined[:, node.children[i].scope] = c[:, node.children[i].scope]
    return joined


def sum_moment(node, children, order=1, evidence=None, dtype=np.float64):
    joined_children = np.array(children)[:, 0, :]
    b = np.array(node.weights, dtype=dtype)
    weighted = np.sum((joined_children * b[:, np.newaxis]), 0, keepdims=True)
    return weighted


def leaf_moment(_node_moment, _node_likelihood):
    def leaf_moment_function(node, order=1, evidence=None):
        leaf_data = evidence[:, node.scope]
        moment_indices = np.isnan(leaf_data)
        shape = evidence.shape
        evidence = evidence[:, node.scope]
        moment = _node_moment(node, order)
        data = np.full(evidence.shape, np.nan)
        data[moment_indices] = moment
        full_data = np.full(shape, np.nan)
        full_data[:, node.scope] = data
        return full_data

    return leaf_moment_function


def Moment(spn, feature_scope, evidence_scope, evidence,
           node_moment=_node_moment, node_likelihoods=_node_likelihood,
           order=1):
    """Compute the moment:

        E[X_feature_scope | X_evidence_scope] given the spn and the evidence data

    Keyword arguments:
    spn -- the spn to compute the probabilities from
    feature_scope -- set() of integers, the scope of the features to get the moment from
    evidence_scope -- set() of integers, the scope of the evidence features
    evidence -- numpy 2d array of the evidence data
    """
    if evidence_scope is None:
        evidence_scope = set()

    if feature_scope is None:
        feature_scope = set(spn.scope)

    assert evidence_scope.intersection(feature_scope) == set(), "Evidence and feature scope must be disjunctive"

    # assert not evidence_scope.union(feature_scope)

    marg_spn = marginalize(spn, feature_scope | evidence_scope)

    node_moments = {Sum: sum_moment, Product: prod_moment}

    for node in get_leaf_types(marg_spn):
        try:
            moment = node_moment[node]
            node_ll = node_likelihoods[node]
        except KeyError:
            raise AssertionError(
                'Node type {} doe not have associated moment and likelihoods'.format(
                    node))
        node_moments[node] = leaf_moment(moment, node_ll)

    fake_evidence = np.full((1, max(spn.scope) + 1), np.nan)

    if evidence is None:
        moment = eval_spn_bottom_up(marg_spn, node_moments, order=order,
                                    evidence=fake_evidence)
        return moment

    all_results = []

    for line in evidence:
        cond_spn = condition(marg_spn, line.reshape(1, -1))
        moment = eval_spn_bottom_up(cond_spn, node_moments, order=order,
                                    evidence=fake_evidence)
        all_results.append(moment)

    return np.array(all_results).reshape(evidence.shape)


def get_mean(spn):
    return Moment(spn, None, None, None)


def get_variance(spn):
    return Moment(spn, None, None, None, order=2) - get_mean(spn) ** 2
