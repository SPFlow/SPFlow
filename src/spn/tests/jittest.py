'''
Created on April 05, 2018

@author: Alejandro Molina
'''
import numpy as np
from numba import jit

from spn.algorithms.Inference import likelihood
from spn.io.Text import str_to_spn
from spn.structure.leaves.Histograms import Histogram

EPSILON = 0.000000000000001

@jit("float64[:](float64[:], float64[:], float64[:], boolean)", nopython=True)
def histogram_likelihood(breaks, densities, data, log_space=True):
    probs = np.zeros((data.shape[0]))

    for i, x in enumerate(data):
        if x < breaks[0] or x > breaks[-1]:
            continue

        # TODO: binary search
        j = 0
        for b in breaks:
            if b > x:
                break
            j += 1

        probs[i] = densities[j - 1]

    probs[probs < EPSILON] = EPSILON

    if log_space:
        return np.log(probs)

    return probs



if __name__ == '__main__':

    n = str_to_spn("""
    (
    Histogram(W1|[ 0., 1., 2.];[0.3, 0.7])
    *
    Histogram(W2|[ 0., 1., 2.];[0.3, 0.7])
    )    
    """, ["W1", "W2"] )

    print(n)

    data = np.hstack(( np.asarray([0.5,0.5,1.5]), np.asarray([0.5,0.5,1.5]) ))

    print(data)

    def hist_ll(node, data, log_space, dtype, node_likelihood):
        return histogram_likelihood(np.array(node.breaks), np.array(node.densities), data, log_space)

    ll = likelihood(n, data, node_likelihood={Histogram: hist_ll})

    print(ll.shape, ll)