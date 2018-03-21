'''
Created on March 21, 2018

@author: Alejandro Molina
'''
import sys
from scipy.misc import logsumexp

from src.spn.structure.Base import Product, Sum
import numpy as np

EPSILON = 0.000000000000001

likelihood_lambdas = {}

def likelihood(node, data):

    if isinstance(node, Product):
        llchildren = np.array([likelihood(c, data) for c in node.children])

        return np.sum(llchildren, axis=0)

    if isinstance(node, Sum):
        llchildren = np.array([likelihood(c, data) for c in node.children])

        weights = np.array(node.weights).reshape(-1, 1)

        return logsumexp(llchildren, b=weights, axis=0)

    tnode = type(node)
    if tnode in likelihood_lambdas:
        return likelihood_lambdas[tnode](node, data)

    raise Exception('Node type not registered: ' + str(type(node)))