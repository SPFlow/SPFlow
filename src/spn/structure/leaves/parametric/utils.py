'''
Created on April 29, 2018

@author: Alejandro Molina
'''
from scipy.stats import *

from spn.structure.leaves.parametric.Parametric import *


def get_scipy_obj_params(node):
    if isinstance(node, Gaussian):
        return norm, {"loc": node.mean, "scale": node.stdev}

    elif isinstance(node, Gamma):
        return gamma, {"a": node.alpha, "scale": 1.0 / node.beta}

    elif isinstance(node, LogNormal):
        return lognorm, {"scale": np.exp(node.mean), "s": node.stdev}

    elif isinstance(node, Poisson):
        return poisson, {"mu": node.mean}

    elif isinstance(node, Geometric):
        return geom, {"p": node.p}


    elif isinstance(node, Exponential):
        return expon, {"scale": 1 / node.l}

    elif isinstance(node, Bernoulli):
        return bernoulli, {"p": node.p}

    else:
        raise Exception("unknown node type %s " % type(node))
