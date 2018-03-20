'''
Created on March 20, 2018

@author: Alejandro Molina
'''
from src.spn.structure.Base import Leaf


class PiecewiseLinear(Leaf):
    def __init__(self, x_range, y_range):
        Leaf.__init__(self)
        self.x_range = x_range
        self.y_range = y_range


class IsotonicUnimodal(PiecewiseLinear):
    def __init__(self, x_range, y_range):
        PiecewiseLinear.__init__(self, x_range, y_range)


class Histogram(Leaf):
    def __init__(self, breaks, densities):
        Leaf.__init__(self)
        self.breaks = breaks
        self.densities = densities