'''
Created on May 22, 2018

@author: Moritz
'''

import numpy as np


def sample_identity_node(node, n_samples, rand_gen=None, ranges=None):
    
    if ranges is None or ranges[node.scope[0]] is None:
        return rand_gen.choice(node.vals, n_samples)
    else:
        #Generate bins for the specified range
        rang = ranges[node.scope[0]]
        
        #Iterate over the specified ranges
        intervals = rang.get_ranges()
        probs = np.zeros(len(intervals))
        bin_vals = []
        for i, interval in enumerate(intervals):
            
            if len(interval) == 1:
                lower = np.searchsorted(node.vals, interval[0], side='left')
                higher = np.searchsorted(node.vals, interval[0], side='right')
            else:
                lower = np.searchsorted(node.vals, interval[0], side='left')
                higher = np.searchsorted(node.vals, interval[1], side='right')
                
            probs[i] = (higher-lower) / len(node.vals)
            bin_vals.append(node.vals[lower:higher])
        
        probs /= np.sum(probs)
        
        #samples = []
        #choices = np.arange(len(bin_vals))
        #while len(samples) < n_samples:
        #    rand_bin = rand_gen.choice(choices, p=probs)
        #    bin_val = bin_vals[rand_bin]
        #    samples.append(rand_gen.choice(bin_val))
            
        insts = probs * n_samples
        insts = np.round(insts)
        
        samples = []
        for i, inst in enumerate(insts):        
            samples = np.concatenate([samples, rand_gen.choice(bin_vals[i], int(inst))])
        rand_gen.shuffle(samples)
            
        return samples
    
    
    
    