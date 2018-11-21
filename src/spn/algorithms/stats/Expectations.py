"""
@author Alejandro Molina
@author Claas Völcker
"""
from spn.algorithms.stats.Moments import Moment, ConditionalMoment, \
    _node_moment


def Expectation(spn, feature_scope=None, evidence=None, node_moment=_node_moment):
    if evidence is not None:
        return ConditionalMoment(spn, evidence, feature_scope,
                          node_moment=node_moment)
    else:
        return Moment(spn, feature_scope, node_moment=node_moment, order=1)
