'''
Created on May 22, 2018

@author: Moritz
'''


import numpy as np

from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear

from spn.experiments.AQP.Ranges import NumericRange


def sample_piecewise_node(node, n_samples, rand_gen, ranges=None):
    assert isinstance(node, PiecewiseLinear)
    assert n_samples > 0
    
    x_range = np.array(node.x_range)
    y_range = np.array(node.y_range)
    
    if ranges is None or ranges[node.scope[0]] is None:
        #Generate bins for random sampling because no range is specified
        bins_x = list(zip(x_range[:-1], x_range[1:]))
        bins_y = list(zip(y_range[:-1], y_range[1:]))
    else:
        #Generate bins for the specified range
        rang = ranges[node.scope[0]]
        assert isinstance(rang, NumericRange)
        
        bins_x = []
        bins_y = []
        
        #Iterate over the specified ranges
        intervals = rang.get_ranges()
        for interval in intervals:
            
            lower = interval[0]
            higher = interval[0] if len(interval) == 1 else interval[1]
            
            lower_prob = np.interp(lower, xp=x_range, fp=y_range)
            higher_prob = np.interp(higher, xp=x_range, fp=y_range)
            indicies = np.where((lower < x_range) & (x_range < higher))
            
            x_interval = [lower] + list(x_range[indicies]) + [higher]
            y_interval = [lower_prob] + list(y_range[indicies]) + [higher_prob]
            
            bins_x += list(zip(x_interval[:-1], x_interval[1:]))
            bins_y += list(zip(y_interval[:-1], y_interval[1:]))
        
    #Compute masses
    masses = []
    for i in range(len(bins_x)):
        if bins_x[i][0] == bins_x[i][1]:
            #Case that the range only contains one value .. Is that correct?
            assert bins_y[i][0] == bins_y[i][1]
            masses.append(bins_y[i][0])
        else:
            masses.append(np.trapz(bins_y[i], bins_x[i]))
    
    #Normalize masses
    masses = np.array(masses) / sum(masses)
    
    return _rejection_sampling(masses, bins_x, bins_y, n_samples, rand_gen)   



def _rejection_sampling(masses, bins_x, bins_y, n_samples, rand_gen):
    
    samples = []
    while len(samples) < n_samples:
         
        rand_bin = rand_gen.choice(len(masses), p=masses)
        #
        # generate random point uniformly in the box
        r_x = rand_gen.uniform(bins_x[rand_bin][0], bins_x[rand_bin][1])
        r_y = rand_gen.uniform(0, bins_y[rand_bin][1])
        #
        # is it in the trapezoid?
        trap_y = np.interp(r_x, xp=bins_x[rand_bin], fp=bins_y[rand_bin])
        if r_y < trap_y:
            samples.append(r_x)
    
    return np.array(samples)

