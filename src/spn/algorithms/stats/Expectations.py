"""
@author Alejandro Molina
@author Claas Völcker
"""
from spn.algorithms.stats.Moments import Moment, ConditionalMoment, _node_moment


def Expectation(spn, feature_scope=None, evidence=None):
    """
    Wrapper function for the moment computation which returns the first moment
    :param spn: a valid spn
    :param feature_scope: optional list of features for which to compute the moments
    :param evidence: optional np array of evidence
    :return:
    """
    if evidence is not None:
        return ConditionalMoment(spn, evidence, feature_scope)
    else:
        return Moment(spn, feature_scope)
