'''
Created on October 19, 2018

@author: Nicola Di Mauro
'''

from spn.algorithms.Inference import add_node_likelihood

def cltree_likelihood(node, data=None, dtype=np.float64):
    raise ValueError('Not Implemented')

def add_parametric_inference_support():
    add_node_likelihood(CLTree, cltree_likelihood)
