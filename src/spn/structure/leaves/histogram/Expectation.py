'''
Created on April 15, 2018

@author: Alejandro Molina
'''

import numpy as np

from spn.structure.StatisticalTypes import MetaType


def histogram_expectation(node, ds_context):
    meta_type = ds_context.meta_types[node.scope[0]]


    exp = 0
    for i in range(len(node.breaks) - 1):
        a = node.breaks[i]
        b = node.breaks[i + 1]
        d = node.densities[i]
        if meta_type == MetaType.DISCRETE:
            sum_x = a
        else:
            sum_x = (b ** 2 - a ** 2) / 2 # integral of x dx, from a to b

        exp += d * sum_x

    return exp



def add_histogram_expectation_support():
    add_node_expectation(Histogram, histogram_expectation)

