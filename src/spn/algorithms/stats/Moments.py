import numpy as np

from spn.algorithms.Inference import likelihood, _node_likelihood
from spn.algorithms.Marginalization import marginalize
from spn.structure.Base import Leaf, get_nodes_by_type, Sum, Product, eval_spn_bottom_up

_node_moment = {}


def add_node_moment(node_type, lambda_func):
    _node_moment[node_type] = lambda_func


def prod_moment(node, children, order=1, evidence=None, dtype=np.float64):
    joined = np.zeros((1, len(node.scope)))
    joined[:] = np.nan
    for i, c in enumerate(children):
        joined[:, node.children[i].scope] = c
    return joined


def sum_moment(node, children, order=1, evidence=None, dtype=np.float64):
    llchildren = np.array(children).reshape(len(children), len(node.scope))
    b = np.array(node.weights, dtype=dtype).reshape(1, -1)
    ret_val = np.dot(b, llchildren)
    return ret_val


def leaf_moment(_node_moment, _node_likelihood):
    def leaf_moment_function(node, order=1, evidence=None):
        shape = evidence.shape
        evidence = evidence[:, node.scope]
        moment_ids = np.isnan(evidence)
        moment = _node_moment(node, order)
        data = np.full(evidence.shape, np.nan)
        data[moment_ids] = moment
        data[~moment_ids] = _node_likelihood(node, evidence[~moment_ids])
        full_data = np.full(shape, np.nan)
        full_data[:,node.scope] = data
        return full_data
    return leaf_moment_function


def Moment(spn, evidence, node_moment=_node_moment, node_likelihoods=_node_likelihood, order=1):
    """Compute the moment:

        E[X_feature_scope | X_evidence_scope] given the spn and the evidence data

    Keyword arguments:
    spn -- the spn to compute the probabilities from
    feature_scope -- set() of integers, the scope of the features to get the moment from
    evidence_scope -- set() of integers, the scope of the evidence features
    evidence -- numpy 2d array of the evidence data
    """

    node_moments = {Sum: sum_moment, Product: prod_moment}

    for node in node_moment.keys():
        try:
            moment = node_moment[node]
            node_ll = node_likelihoods[node]
        except KeyError:
            raise AssertionError('Node type {} doe not have associated moment and likelihoods'.format(node))
        node_moments[key] = leaf_moment(moment, node_ll)

    if evidence is None:
        # fake_evidence needs to be computed
        evidence = np.full((1, len(spn.scope)), np.nan)

    moment = eval_spn_bottom_up(spn, node_moments, order=order, evidence=evidence)
    prob = likelihood(spn, evidence)
    
    return moment/prob


def get_mean(spn):
    return Moment(spn, None)


def get_variance(spn):
    return Moment(spn, None, order=2) - get_mean(spn) ** 2
