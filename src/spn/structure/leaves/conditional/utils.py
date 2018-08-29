'''
Created on April 29, 2018

@author: Alejandro Molina
'''
from scipy.stats import *

from spn.structure.leaves.conditional.Conditional import *


def logit(x):
    return np.exp(x) / (1 + np.exp(x))


def get_scipy_obj_params(node, obs):
    # w*x + bias
    pred = np.dot(obs, node.weights[:-1]) + node.weights[-1]
    if isinstance(node, Conditional_Gaussian):
        mean = pred
        return norm, {"loc": mean, "scale": np.ones(obs.shape[0])*0.01}

    elif isinstance(node, Conditional_Poisson):
        mu = np.exp(pred)
        return poisson, {"mu": mu}

    elif isinstance(node, Conditional_Bernoulli):
        p = logit(pred)
        return bernoulli, {"p": p}

    else:
        raise Exception("unknown node type %s " % type(node))
