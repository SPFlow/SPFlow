'''
Created on June 21, 2018

@author: Moritz
'''

from spn.algorithms.stats.Expectations import add_node_expectation
from spn.experiments.AQP.leaves.static.StaticNumeric import StaticNumeric


def static_expectation(node):
    return node.val


def add_piecewise_expectation_support():
    add_node_expectation(StaticNumeric, static_expectation)
