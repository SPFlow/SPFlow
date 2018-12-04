"""
Created on April 15, 2018

@author: Alejandro Molina
"""

import numpy as np

from spn.algorithms.stats.Moments import add_node_moment
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear


def piecewise_moment(node, order=1):
    exp = 0
    for i in range(len(node.x_range) - 1):
        y0 = node.y_range[i]
        y1 = node.y_range[i + 1]
        x0 = node.x_range[i]
        x1 = node.x_range[i + 1]

        # compute the line of the top of the trapezoid
        m = (y0 - y1) / (x0 - x1)
        b = -m * x0 + y0
        k = order
        integral = m / (k + 2) * (x1 ** (k + 2) - x0 ** (k + 2)) + b / (k + 1) * (x1 ** (k + 1) - x0 ** (k + 1))
        exp += integral
    return exp


def add_piecewise_moment_support():
    add_node_moment(PiecewiseLinear, piecewise_moment)
