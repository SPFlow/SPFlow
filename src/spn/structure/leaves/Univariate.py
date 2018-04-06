'''
Created on March 20, 2018
@author: Alejandro Molina
'''
import numpy as np

from spn.structure.Base import Leaf


class Parametric(Leaf):
    def __init__(self, name, params):
        Leaf.__init__(self)
        self.params = params

class Poisson(Leaf):
    def __init__(self, mean):
        Leaf.__init__(self)
        self.mean = mean

class Normal(Leaf):
    def __init__(self, mean, stdev):
        Leaf.__init__(self)
        self.mean = mean
        self.stdev = stdev

def create_leaf_univariate(data, ds_context, scope):
    assert(len(scope) == 1, "scope of univariate for more than one variable?")
    assert(data.shape[1] == 1, "data has more than one feature?")

    idx = scope[0]

    family = ds_context.family[idx]

    if family == "normal":
        mean = np.mean(data)
        stdev = np.std(data)
        return Normal(mean, stdev)

    if family == "poisson":
        assert (np.all(data >= 0), "poisson negative?")
        mean = np.mean(data)
        return Poisson(mean)

    if family == "bernoulli":
        assert(len(np.unique(data)) != 1, "data not binary for bernoulli")
        p = np.sum(data) / data.shape[0]
        return Bernoulli(p)

    raise Exception('Unknown family: ' + family)

def univariate_to_str(node, feature_names=None):
    if feature_names is None:
        fname = "V"+str(node.scope[0])
    else:
        fname = feature_names[node.scope[0]]

    if isinstance(node, Bernoulli):
        return "Bernoulli(%s|ρ=%s)" % (fname, node.p)

    elif isinstance(node, Poisson):
        return "Poisson(%s|λ=%s)" % (fname, node.mean)

    elif isinstance(node, Normal):
        return "Normal(%s|μ=%s,σ=%s)" % (fname, node.mean, node.stdev)


