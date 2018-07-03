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




def sample_fast_piecewise_node(x_range, y_range, ranges, n_samples, rand_gen):
    
    if ranges is None or ranges[0] is None:
        #Generate bins for random sampling because no range is specified
        bins_x = list(zip(x_range[:-1], x_range[1:]))
        bins_y = list(zip(y_range[:-1], y_range[1:]))
    else:
        #Generate bins for the specified range
        rang = ranges[0]
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
     
    cumulative_stats = []
    for i in range(len(bins_x)):
        y0 = bins_y[i][0]
        y1 = bins_y[i][1]
        x0 = bins_x[i][0]
        x1 = bins_x[i][1]

        m = (y0 - y1) / (x0 - x1)
        b = (x0 * y1 - x1 * y0) / (x0 - x1)
        
        lowest_y = __integral(m,b, x0)
        highest_y = __integral(m,b, x1)
        prob = highest_y - lowest_y
        
        cumulative_stats.append([m, b, x0, x1, lowest_y, highest_y, prob])
    
    cumulative_stats = np.array(cumulative_stats, dtype=np.float64)
    
    cumulative_stats[:,6] = cumulative_stats[:,6] / np.sum(cumulative_stats[:,6])

    rand_probs = rand_gen.rand(n_samples)
    
    vals = [__inverse_cumulative(cumulative_stats, prob) for prob in rand_probs]
    
    return np.array(vals)



EPS = 0.00000001


def __integral(m, b, x):
    return 0.5 * m * (x ** 2) + b * x


def __inverse_integral(m, b, y):
    p = b/ (0.5 * m)
    q = y/(0.5 * m)
    
    #Our scenario is easy we exactly know what value we want
    if m >= 0:
        #Left value is important
        return - (p/2) + np.sqrt((p/2)**2 + q)
    else:
        #Right value is important
        return - (p/2) - np.sqrt((p/2)**2 + q)


def __inverse_cumulative(cumulative_stats, y):
    
    bin_probs = cumulative_stats[:,6]
    cumulative_probs = np.cumsum(bin_probs)
    
    # +EPS to avoid incorrect behavior caused by floating point errors
    cumulative_probs[-1] += EPS
    
    bin_id = np.where(( y <= cumulative_probs))[0][0]
    stats = cumulative_stats[bin_id]
    
    lower_cumulative_prob = 0 if bin_id == 0 else cumulative_probs[bin_id-1]
    y_perc = (y - lower_cumulative_prob) / bin_probs[bin_id]
    y_val = (stats[5] - stats[4]) * y_perc + stats[4]

    return __inverse_integral(stats[0], stats[1], y_val)

