'''
Created on April 15, 2018

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
        #Generate random samples because no range is specified
        bins_x = []
        bins_y = []
        masses = []
        
        tmp_val_x = x_range[0]
        tmp_val_y = y_range[0]
        for i in range(1, len(x_range)):
            mass = (y_range[i] + tmp_val_y)/2. * (x_range[i] - tmp_val_x)
            bins_x.append([tmp_val_x, x_range[i]])
            bins_y.append([tmp_val_y, y_range[i]])
            masses.append(mass)
            
        masses = np.array(masses) / sum(masses)
        
        return _rejection_sampling(masses, bins_x, bins_y, n_samples, rand_gen)
    else:
        #Generate samples for the specified range
        rang = ranges[node.scope[0]]
        assert isinstance(rang, NumericRange)
        
        intervals = rang.get_ranges()
        
        #Generate bins for sampling
        bins_x = []
        bins_y = []
        masses = []
        
        for lower, higher in intervals:
            
            lower_prob = np.interp(lower, xp=x_range, fp=y_range)
            higher_prob = np.interp(higher, xp=x_range, fp=y_range)
            indicies = np.where((lower < x_range) & (x_range < higher))
            
            x_interval = [lower] + list(x_range[indicies]) + [higher]
            y_interval = [lower_prob] + list(y_range[indicies]) + [higher_prob]
            
            tmp_val_x = x_interval[0]
            tmp_val_y = y_interval[0]
            for i in range(1, len(x_interval)):
                mass = (y_interval[i] + tmp_val_y)/2. * (x_interval[i] - tmp_val_x)
                bins_x.append([tmp_val_x, x_interval[i]])
                bins_y.append([tmp_val_y, y_interval[i]])
                masses.append(mass)
                
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