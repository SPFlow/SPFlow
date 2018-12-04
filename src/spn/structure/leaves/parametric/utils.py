"""
Created on April 29, 2018

@author: Alejandro Molina
"""
from scipy.stats import *

from spn.structure.leaves.parametric.Parametric import *


def get_scipy_obj_params(node):
    if isinstance(node, Gaussian):
        assert node.mean is not None
        assert node.stdev is not None
        return norm, {"loc": node.mean, "scale": node.stdev}

    elif isinstance(node, Gamma):
        assert node.alpha is not None
        assert node.beta is not None
        return gamma, {"a": node.alpha, "scale": 1.0 / node.beta}

    elif isinstance(node, LogNormal):
        assert node.mean is not None
        assert node.stdev is not None
        return lognorm, {"scale": np.exp(node.mean), "s": node.stdev}

    elif isinstance(node, Poisson):
        assert node.mean is not None
        return poisson, {"mu": node.mean}

    elif isinstance(node, Geometric):
        assert node.p is not None
        return geom, {"p": node.p}

    elif isinstance(node, Exponential):
        assert node.l is not None
        return expon, {"scale": 1 / node.l}

    elif isinstance(node, Bernoulli):
        assert node.p is not None
        return bernoulli, {"p": node.p}

    else:
        raise Exception("unknown node type %s " % type(node))
