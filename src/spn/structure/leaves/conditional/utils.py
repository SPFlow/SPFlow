'''
Created on April 29, 2018

@author: Alejandro Molina
'''
from scipy.stats import *

from spn.structure.leaves.conditional.Conditional import *


def get_scipy_obj_params(node, obs):
    if isinstance(node, Conditional_Gaussian):
        mean = node.inv_linkfunc(np.dot(obs, node.weights))
        assert len(mean) == obs.shape[0]
        # assert len(node.mean) == obs.shape[0]
        # assert len(node.stdev) == obs.shape[0]
        return norm, {"loc": mean, "scale": node.stdev}  # should be a vector, instead of a scalar

    elif isinstance(node, Conditional_Poisson):
        # assert len(node.mean) == obs.shape[0]
        mu = node.inv_linkfunc(np.dot(obs, node.weights))
        assert len(mu) == obs.shape[0]
        return poisson, {"mu": mu}

    elif isinstance(node, Conditional_Bernoulli):
        # assert len(node.p) == obs.shape[0]
        p = node.inv_linkfunc(np.dot(obs, node.weights))
        assert len(p) == obs.shape[0]
        return bernoulli, {"p": p}

    else:
        raise Exception("unknown node type %s " % type(node))
