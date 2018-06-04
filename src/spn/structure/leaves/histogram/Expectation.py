'''
Created on April 15, 2018

@author: Alejandro Molina
'''

import numpy as np

from spn.algorithms.stats.Expectations import add_node_expectation
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Histograms import Histogram


def histogram_expectation(node):

    exp = 0
    for i in range(len(node.breaks) - 1):
        a = node.breaks[i]
        b = node.breaks[i + 1]
        d = node.densities[i]
        if node.meta_type == MetaType.DISCRETE:
            sum_x = a
        else:
            sum_x = (b ** 2 - a ** 2) / 2 # integral of x dx, from a to b

        exp += d * sum_x

    return exp


def add_histogram_expectation_support():
    add_node_expectation(Histogram, histogram_expectation)
