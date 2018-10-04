'''
Created on April 15, 2018

@author: Alejandro Molina
'''

import numpy as np

from spn.algorithms.Inference import EPSILON, add_node_likelihood
from spn.structure.leaves.histogram.Histograms import Histogram
#from numba import jit
import bisect

#@jit("float64[:](float64[:], float64[:], float64[:,:])", nopython=True)
def histogram_ll(breaks, densities, data):


    probs = np.zeros((data.shape[0], 1))

    for i, x in enumerate(data):
        if x < breaks[0] or x >= breaks[-1]:
            continue

        probs[i] = densities[bisect.bisect(breaks, x) - 1]

        # j = 0
        # for b in breaks:
        #    if b > x:
        #        break
        #    j += 1
        #
        # probs[i] = densities[j - 1]

    probs[probs < EPSILON] = EPSILON

    return probs


def histogram_likelihood(node, data=None, dtype=np.float64):
    probs = np.ones((data.shape[0], 1), dtype=dtype)

    nd = data[:, node.scope[0]]
    marg_ids = np.isnan(nd)

    probs[~marg_ids] = histogram_ll(np.array(node.breaks), np.array(node.densities), nd[~marg_ids])

    return probs



def add_histogram_inference_support():
    add_node_likelihood(Histogram, histogram_likelihood)
