'''
Created on March 20, 2018

@author: Alejandro Molina
'''
from src.spn.structure.Base import Leaf


class Bernoulli(Leaf):
    def __init__(self, p):
        Leaf.__init__(self)
        self.p = p

class Poisson(Leaf):
    def __init__(self, mean):
        Leaf.__init__(self)
        self.mean = mean

class Normal(Leaf):
    def __init__(self, mean, stdev):
        Leaf.__init__(self)
        self.mean = mean
        self.stdev = stdev