'''
Created on March 21, 2018

@author: Alejandro Molina
'''
import sys
from scipy.misc import logsumexp

from spn.structure.Base import Product, Sum, Leaf
import numpy as np

EPSILON = 0.000000000000001


def log_likelihood(node, data, leaf_likelihood=None):

    if isinstance(node, Product):
        llchildren = np.zeros((data.shape[0], len(node.children)))

        #TODO: parallelize here
        for i, c in enumerate(node.children):
            llchildren[:,i] = log_likelihood(c, data, leaf_likelihood)

        return np.sum(llchildren, axis=1)

    if isinstance(node, Sum):
        llchildren = np.zeros((data.shape[0], len(node.children)))

        # TODO: parallelize here
        for i, c in enumerate(node.children):
            llchildren[:,i] = log_likelihood(c, data, leaf_likelihood)

        b = np.array(node.weights).reshape(1, -1)

        return logsumexp(llchildren, b=b, axis=1)

    if isinstance(node, Leaf):
        return leaf_likelihood(node, data[:, node.scope])

    raise Exception('Node type not registered: ' + str(type(node)))
