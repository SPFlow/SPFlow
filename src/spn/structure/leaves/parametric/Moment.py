"""
Created on April 15, 2018

@author: Alejandro Molina
"""
from spn.algorithms.stats.Moments import add_node_moment
from spn.structure.leaves.parametric.Parametric import *
import numpy as np


def parametric_moment(node, order=1):
    if order > 1:
        return NotImplementedError("Higher moments are not implemented yet")

    if isinstance(node, Gaussian) or isinstance(node, Poisson):
        return node.mean

    elif isinstance(node, Uniform):
        return (node.start + node.end) / 2

    elif isinstance(node, Gamma):
        return ValueError("Not Implemented")

    elif isinstance(node, LogNormal):
        return np.exp(node.mean)

    elif isinstance(node, Bernoulli):
        return node.p

    elif isinstance(node, NegativeBinomial):
        return node.p * node.n / (1 - node.p)

    elif isinstance(node, Hypergeometric):
        return node.n * (node.K / node.N)

    elif isinstance(node, Geometric):
        return -1 / np.log2(1 - node.p)

    elif isinstance(node, Categorical):
        raise ValueError("Not Implemented")

    elif isinstance(node, Exponential):
        return 1 / node.l
    else:
        raise Exception("Unknown parametric " + str(type(node)))


def add_parametric_moment_support():
    add_node_moment(Gaussian, parametric_moment)
    add_node_moment(Gamma, parametric_moment)
    add_node_moment(LogNormal, parametric_moment)
    add_node_moment(Poisson, parametric_moment)
    add_node_moment(Bernoulli, parametric_moment)
    add_node_moment(Categorical, parametric_moment)
    add_node_moment(NegativeBinomial, parametric_moment)
    add_node_moment(Hypergeometric, parametric_moment)
    add_node_moment(Geometric, parametric_moment)
    add_node_moment(Exponential, parametric_moment)
    add_node_moment(Uniform, parametric_moment)
