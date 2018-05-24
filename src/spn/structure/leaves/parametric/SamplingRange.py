'''
Created on May 2, 2018

@author: Moritz
'''

import numpy as np

from spn.experiments.AQP.Ranges import NominalRange
from spn.structure.leaves.parametric.Parametric import Categorical


def sample_categorical_node(node, n_samples, rand_gen, ranges=None):
    assert isinstance(node, Categorical)
    assert n_samples > 0
    
    if ranges is None or ranges[node.scope[0]] is None:
        #Generate random samples because no range is specified
        return rand_gen.choice(np.arange(node.k), p=node.p, size=n_samples)
    else:
        #Generate samples for the specified range
        rang = ranges[node.scope[0]]
        assert isinstance(rang, NominalRange)
        
        possible_vals = rang.get_ranges()        
        probabilities = np.array(node.p)
        
        possible_probs = probabilities[possible_vals]/(np.sum(probabilities[possible_vals]))
        
        return rand_gen.choice(possible_vals, p=possible_probs, size=n_samples)
