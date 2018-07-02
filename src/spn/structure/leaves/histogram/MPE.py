'''
Created on July 02, 2018

@author: Alejandro Molina
'''
from spn.algorithms import add_node_mpe, mpe_leaf
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear


def add_parametric_mpe_support():
    add_node_mpe(PiecewiseLinear, mpe_leaf)

