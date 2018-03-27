'''
Created on March 21, 2018

@author: Alejandro Molina
'''
import numpy as np
from scipy.misc import logsumexp

from spn.structure.Base import Product, Sum, Leaf

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

    raise Exception('Node type unknown: ' + str(type(node)))

def conditional_log_likelihood(node_joint, node_marginal, data, leaf_likelihood=None):
    return log_likelihood(node_joint, data, leaf_likelihood) - log_likelihood(node_marginal, data, leaf_likelihood)