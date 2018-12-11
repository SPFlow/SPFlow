"""
Created on April 29, 2018

@author: Alejandro Molina
"""
from scipy.stats import *

from spn.structure.leaves.parametric.Parametric import *


def get_scipy_obj(param_type):
    if param_type == Gaussian:
        return norm

    elif param_type == Gamma:
        return gamma

    elif param_type == LogNormal:
        return lognorm

    elif param_type == Poisson:
        return poisson

    elif param_type == Geometric:
        return geom

    elif param_type == Exponential:
        return expon

    elif param_type == Bernoulli:
        return bernoulli

    else:
        raise Exception("unknown node type %s " % str(param_type))


def get_scipy_obj_params(node):
    scipy_ob = get_scipy_obj(type(node))

    if isinstance(node, Gaussian):
        assert node.mean is not None
        assert node.stdev is not None
        params = {"loc": node.mean, "scale": node.stdev}

    elif isinstance(node, Gamma):
        assert node.alpha is not None
        assert node.beta is not None
        params = {"a": node.alpha, "scale": 1.0 / node.beta}

    elif isinstance(node, LogNormal):
        assert node.mean is not None
        assert node.stdev is not None
        params = {"scale": np.exp(node.mean), "s": node.stdev}

    elif isinstance(node, Poisson):
        assert node.mean is not None
        params = {"mu": node.mean}

    elif isinstance(node, Geometric):
        assert node.p is not None
        params = {"p": node.p}

    elif isinstance(node, Exponential):
        assert node.l is not None
        params = {"scale": 1 / node.l}

    elif isinstance(node, Bernoulli):
        assert node.p is not None
        params = {"p": node.p}

    else:
        raise Exception("unknown node type %s " % type(node))

    return scipy_ob, params
