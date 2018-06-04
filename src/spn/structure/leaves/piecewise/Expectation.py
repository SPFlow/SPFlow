'''
Created on April 15, 2018

@author: Alejandro Molina
'''

import numpy as np

from spn.algorithms.stats.Expectations import add_node_expectation
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear


def piecewise_expectation(node):
    exp = 0
    for i in range(len(node.x_range) - 1):
        y0 = node.y_range[i]
        y1 = node.y_range[i + 1]
        x0 = node.x_range[i]
        x1 = node.x_range[i + 1]

        # compute the line of the top of the trapezoid
        m = (y0 - y1) / (x0 - x1)
        b = (x0 * y1 - x1 * y0) / (x0 - x1)

        # integral from w to z, of x * (mx+b) dx
        w = x0
        z = x1
        integral = (1 / 6) * (-3 * b * (w ** 2) + 3 * b * (z ** 2) - 2 * m * (w ** 3) + 2 * m * (z ** 3))
        exp += integral

    return exp


def add_piecewise_expectation_support():
    add_node_expectation(PiecewiseLinear, piecewise_expectation)
