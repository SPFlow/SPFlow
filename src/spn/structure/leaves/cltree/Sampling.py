'''
Created on October 22, 2018

@author: Nicola Di Mauro
@author: Antonio Vergari
'''
from spn.algorithms.Sampling import add_node_sampling
from spn.structure.leaves.cltree.CLTree import CLTree

def sample_cltree_node(node, n_samples, data, rand_gen):
    raise Exception('Not implemented')

def add_cltree_sampling_support():
    add_node_sampling(CLTree, sample_cltree_node)

