'''
Created on April 15, 2018

@author: Alejandro Molina
'''
from spn.algorithms.stats.Expectations import add_node_expectation
from spn.structure.leaves.parametric.Parametric import *
import numpy as np


def parametric_expectation(node):
    if isinstance(node, Gaussian) or isinstance(node, Poisson):
        return node.mean

    elif isinstance(node, Uniform):
        return (node.start + node.end) / 2

    elif isinstance(node, Gamma):
        return ValueError('Not Implemented')

    elif isinstance(node, LogNormal):
        return np.exp(node.mean)

    elif isinstance(node, Bernoulli):
        return node.p

    elif isinstance(node, NegativeBinomial):
        return node.p * node.n / (1 - node.p)

    elif isinstance(node, Hypergeometric):
        return node.n * (node.K / node.N)


    elif isinstance(node, Geometric):
        return -1 / np.log2(1-node.p)

    elif isinstance(node, Categorical):
        raise ValueError('Not Implemented')

    elif isinstance(node, Exponential):
        return 1 / node.l
    else:
        raise Exception("Unknown parametric " + str(type(node)))


def add_parametric_expectation_support():
    add_node_expectation(Gaussian, parametric_expectation)
    add_node_expectation(Gamma, parametric_expectation)
    add_node_expectation(LogNormal, parametric_expectation)
    add_node_expectation(Poisson, parametric_expectation)
    add_node_expectation(Bernoulli, parametric_expectation)
    add_node_expectation(Categorical, parametric_expectation)
    add_node_expectation(NegativeBinomial, parametric_expectation)
    add_node_expectation(Hypergeometric, parametric_expectation)
    add_node_expectation(Geometric, parametric_expectation)
    add_node_expectation(Exponential, parametric_expectation)
    add_node_expectation(Uniform, parametric_expectation)
