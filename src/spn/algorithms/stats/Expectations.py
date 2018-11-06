"""
@author Alejandro Molina
@author Claas Völcker
"""
from spn.algorithms.stats.Moments import Moment, _node_moment


def Expectation(spn, evidence, node_moment=_node_moment):
    return Moment(spn, evidence, node_moment=_node_moment, order=1)
