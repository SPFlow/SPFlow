'''
Created on April 15, 2018

@author: Alejandro Molina
'''
from spn.algorithms.stats.Expectations import add_node_expectation
from spn.structure.leaves.conditional.Conditional import *
import numpy as np


def conditional_expectation(node):
    if isinstance(node, Conditional_Gaussian) or isinstance(node, Conditional_Poisson):
        return node.mean

    elif isinstance(node, Conditional_Bernoulli):
        return node.p

    else:
        raise Exception("Unknown parametric " + str(type(node)))


def add_parametric_expectation_support():
    add_node_expectation(Conditional_Gaussian, conditional_expectation)
    add_node_expectation(Conditional_Poisson, conditional_expectation)
    add_node_expectation(Conditional_Bernoulli, conditional_expectation)
