'''
Created on May 22, 2018

@author: Moritz
'''

import numpy as np

from spn.algorithms.Inference import add_node_likelihood
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear


def piecewise_likelihood_range(node, ranges, dtype=np.float64, **kwargs):
    '''
    Returns the probability for the given ranges.
    
    ranges is multi-dimensional array:
    - First index specifies the instance
    - Second index specifies the feature
    
    Each entry of range contains a Range-object or None (e.g. for piecewise-node NumericRange exists).
    If the entry is None, then the log-likelihood probability of 0 will be returned.
    '''
    
    #Assert context is not None and assert that the given node is only build on one instance
    assert len(node.scope) == 1, node.scope
    
    #Initialize the return variable log_probs with zeros
    probs = np.ones((ranges.shape[0], 1), dtype=dtype)
    
    #Only select the ranges for the specific feature
    ranges = ranges[:, node.scope[0]]
    
    
    for i, rang in enumerate(ranges):
        
        #Skip if no range is specified aka use a log-probability of 0 for that instance
        if rang is None:
            continue
        
        #Skip if no values for the range are provided
        if rang.is_impossible():
            probs[i] = 0
        
        #Compute the sum of the probability of all possible values
        probs[i] = sum([_compute_probability_for_range(node, interval) for interval in rang.get_ranges()])
        
    return probs

    
    
def _compute_probability_for_range(node, interval):
    
    if len(interval) == 1:
        return np.interp(x=interval[0], xp=node.x_range, fp=node.y_range)
    
    lower = interval[0]
    higher = interval[1]
    
    x_range = np.array(node.x_range)
    y_range = np.array(node.y_range)
    
    if lower <= x_range[0] and higher >= x_range[-1]:
        return 1.
    
    lower_prob = np.interp(lower, xp=x_range, fp=y_range)
    higher_prob = np.interp(higher, xp=x_range, fp=y_range)
    
    indicies = np.where((lower < x_range) & (x_range < higher))
    
    x_tmp = [lower] + list(x_range[indicies]) + [higher]
    y_tmp = [lower_prob] + list(y_range[indicies]) + [higher_prob]
    
    return np.trapz(y_tmp, x_tmp)



def add_piecewise_inference_range_support():
    add_node_likelihood(PiecewiseLinear, piecewise_likelihood_range)