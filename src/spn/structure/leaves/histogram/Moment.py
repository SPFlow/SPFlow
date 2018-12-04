"""
Created on April 15, 2018

@author: Alejandro Molina
@author: Claas Völcker
"""

import numpy as np

from spn.algorithms.stats.Moments import add_node_moment
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Histograms import Histogram


def histogram_moment(node, order=1):

    exp = 0
    for i in range(len(node.breaks) - 1):
        a = node.breaks[i]
        b = node.breaks[i + 1]
        d = node.densities[i]
        if node.meta_type == MetaType.DISCRETE:
            sum_x = a ** order
        else:
            sum_x = (b ** (order + 1) - a ** (order + 1)) / (order + 1)

        exp += d * sum_x
    return exp


def add_histogram_moment_support():
    add_node_moment(Histogram, histogram_moment)
