'''
Created on May 22, 2018

@author: Moritz
'''

import numpy as np

from spn.algorithms.Inference import add_node_likelihood
from spn.structure.leaves.parametric.Parametric import Categorical


def categorical_likelihood_range(node, ranges, dtype=np.float64, **kwargs):
    '''
    Returns the probability for the given ranges.
    
    ranges is multi-dimensional array:
    - First index specifies the instance
    - Second index specifies the feature
    
    Each entry of range contains a Range-object or None (e.g. for categorical-node NominalRange exists).
    If the entry is None, then the log-likelihood probability of 0 will be returned.
    '''
    
    #Assert that the given node is only build on one instance
    assert len(node.scope) == 1, node.scope
    
    #Initialize the return variable log_probs with zeros
    probs = np.ones((ranges.shape[0], 1), dtype=dtype)
    
    #Only select the ranges for the specific feature
    ranges = ranges[:, node.scope[0]]
    
    #For each instance
    for i, rang in enumerate(ranges):
        
        #Skip if no range is specified aka use a log-probability of 0 for that instance
        if rang is None:
            continue
        
        #Skip if no values for the range are provided
        if rang.is_impossible():
            probs[i] = 0
        
        #Compute the sum of the probability of all possible values
        probs[i] = sum([node.p[possible_val] for possible_val in rang.get_ranges()])
            
    return probs
    


def add_parametric_inference_range_support():
    add_node_likelihood(Categorical, categorical_likelihood_range)

