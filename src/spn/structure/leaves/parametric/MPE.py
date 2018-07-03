'''
Created on July 02, 2018

@author: Alejandro Molina
'''
from spn.algorithms import add_node_mpe, mpe_leaf
from spn.structure.leaves.parametric.Parametric import Gaussian, Gamma, LogNormal, Poisson, Bernoulli, Categorical, \
    NegativeBinomial, Hypergeometric, Geometric, Exponential


def add_parametric_mpe_support():
    add_node_mpe(Gaussian, mpe_leaf)
    add_node_mpe(Gamma, mpe_leaf)
    add_node_mpe(LogNormal, mpe_leaf)
    add_node_mpe(Poisson, mpe_leaf)
    add_node_mpe(Bernoulli, mpe_leaf)
    add_node_mpe(Categorical, mpe_leaf)
    add_node_mpe(NegativeBinomial, mpe_leaf)
    add_node_mpe(Hypergeometric, mpe_leaf)
    add_node_mpe(Geometric, mpe_leaf)
    add_node_mpe(Exponential, mpe_leaf)
