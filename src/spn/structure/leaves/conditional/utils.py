'''
Created on April 29, 2018

@author: Alejandro Molina
'''
from scipy.stats import *

from spn.structure.leaves.conditional.Conditional import *


def get_scipy_obj_params(node, obs):
    if isinstance(node, Conditional_Gaussian):
        assert len(node.mean) == obs.shape[0]
        assert len(node.stdev) == obs.shape[0]
        return norm, {"loc": node.mean, "scale": node.stdev}  # should be a vector, instead of a scalar

    elif isinstance(node, Conditional_Poisson):
        assert len(node.mean) == obs.shape[0]
        return poisson, {"mu": node.mean}

    elif isinstance(node, Conditional_Bernoulli):
        assert len(node.p) == obs.shape[0]
        return bernoulli, {"p": node.p}

    else:
        raise Exception("unknown node type %s " % type(node))
