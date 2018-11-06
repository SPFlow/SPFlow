import numpy as np

from spn.algorithms.Inference import likelihood
from spn.algorithms.Marginalization import marginalize
from spn.structure.Base import Leaf, get_nodes_by_type, Sum, Product, eval_spn_bottom_up

_node_moment = {}


def add_node_moment(node_type, lambda_func):
    _node_moment[node_type] = lambda_func


def prod_moment(node, children, order=1, dtype=np.float64):
    joined = np.zeros((1, len(node.scope)))
    joined[:] = np.nan
    for i, c in enumerate(children):
        joined[:, node.children[i].scope] = c
    return joined


def sum_moment(node, children, order=1, dtype=np.float64):
    llchildren = np.array(children).reshape(len(children), len(node.scope))
    b = np.array(node.weights, dtype=dtype).reshape(1, -1)
    ret_val = np.dot(b, llchildren)
    return ret_val


def Moment(spn, feature_scope, evidence_scope, evidence, node_moment=_node_moment, order=1):
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

    assert not (len(evidence_scope) > 0 and evidence is None)

    assert len(feature_scope.intersection(evidence_scope)) == 0

    marg_spn = marginalize(spn, keep=feature_scope | evidence_scope)

    node_moments = _node_moment
    node_moments.update({Sum: sum_moment,
                              Product: prod_moment})

    if evidence is None:
        # fake_evidence is not used
        fake_evidence = np.zeros((1, len(spn.scope))).reshape(1,-1)
        moment = eval_spn_bottom_up(marg_spn, node_moments, order=order)
        return moment

    # if we have evidence, we want to compute the conditional moment
    else:
        raise NotImplementedError('Please use a conditional SPN to calculated conditional moments')

    return moment


def get_mean(spn):
    return Moment(spn, set(spn.scope), None, None)


def get_variance(spn):
    return Moment(spn, set(spn.scope), None, None,
                       order=2) - get_mean(spn) ** 2
