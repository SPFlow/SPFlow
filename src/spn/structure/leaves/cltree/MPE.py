'''
Created on October 22, 2018

@author: Nicola DI Mauro
'''
from spn.algorithms.MPE import get_mpe_top_down_leaf, add_node_mpe
from spn.structure.leaves.cltree.Inference import cltree_likelihood
from spn.structure.leaves.cltree.CLTree import CLTree


def get_cltree_bottom_up_ll(ll_func):
    return 0


def get_cltree_top_down_ll():
    return 0


def add_cltree_mpe_support():
    add_node_mpe(CLTree, get_cltree_bottom_up_ll(cltree_likelihood),
                 get_cltree_top_down_ll())


