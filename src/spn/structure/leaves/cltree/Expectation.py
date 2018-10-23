'''
Created on October 18, 2018

@author: Nicola Di Mauro
'''
from spn.algorithms.stats.Expectations import add_node_expectation
from spn.structure.leaves.cltree.CLTree import *

def cltree_expectation(node):
    raise ValueError('Not Implemented')

def add_cltree_expectation_support():
    add_node_expectation(CLTree, cltree_expectation)
