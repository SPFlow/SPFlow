'''
Created on March 20, 2018
@author: Alejandro Molina
'''

import numpy as np

from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import Type

identity = lambda x: x
exponential = lambda x: np.exp(x)
logit = lambda x: np.exp(x) / (1 + np.exp(x))

class Conditional(Leaf):
    def __init__(self, type, scope=None):
        Leaf.__init__(self, scope=scope)
        self._type = type

    @property
    def type(self):
        return self._type

    @property
    def params(self):
        raise Exception("Not Implemented")


class Conditional_Gaussian(Conditional):
    """
    Implements a conditional univariate gaussian distribution with parameters
    \mu(mean)
    \sigma ^ 2 (variance)
    (alternatively \sigma is the standard deviation(stdev) and \sigma ^ {-2} the precision)
    self.mean is a vector
    """

    # def __init__(self, mean=None, stdev=None, scope=None):
    def __init__(self, inputs=None, params=None, inv_linkfunc=identity, scope=None):
        Conditional.__init__(self, Type.REAL, scope=scope)

        # parameters
        self.weights = params
        self.inv_linkfunc = inv_linkfunc
        self.mean = self.inv_linkfunc(np.dot(inputs, self.weights))
        if inputs is not None and self.weights is not None:
            self.stdev = 1  #todo

    @property
    def params(self):
        return {'mean': self.mean, 'stdev': self.stdev}

    @property
    def precision(self):
        return 1.0 / self.variance

    @property
    def variance(self):
        return self.stdev * self.stdev

    @property
    def mode(self):
        return self.mean


class Conditional_Poisson(Conditional):
    """
    Implements a univariate Poisson distribution with parameter
    \lambda (mean)
    self.mean is a vector
    """

    # def __init__(self, mean=None, scope=None):
    def __init__(self, inputs=None, params=None, inv_linkfunc=exponential, scope=None):
        Conditional.__init__(self, Type.COUNT, scope=scope)

        self.weights = params
        # self._inv_linkfunc = inv_linkfunc
        self.inv_linkfunc = inv_linkfunc
        if inputs is not None and self.weights is not None:
            self.mean = self.inv_linkfunc(np.dot(inputs, self.weights))
            # self.mean = self._inv_linkfunc(np.dot(inputs, self.weights))

    @property
    def params(self):
        return {'mean': self.mean}

    # @property
    # def inv_linkfunc(self):
    #     return self._inv_linkfunc

    @property
    def mode(self):
        return np.floor(self.mean)

class Conditional_Bernoulli(Conditional):
    """
    Implements a univariate Bernoulli distribution with parameter
    p (probability of a success)
    self.p is a list param values
    """

    # def __init__(self, p=None, scope=None):
    def __init__(self, inputs=None, params=None, inv_linkfunc=logit, scope=None):
        Conditional.__init__(self, Type.BINARY, scope=scope)

        self.weights = params
        self.inv_linkfunc = inv_linkfunc
        if inputs is not None and self.weights is not None:
            self.p = self.inv_linkfunc(np.dot(inputs, self.weights))

    @property
    def params(self):
        return {'p': self.p}

    @property
    def mode(self):
        if self.p > 0:
            return 1
        else:
            return 0


def create_conditional_leaf(data, ds_context, scope):
    from spn.structure.leaves.conditional.MLE import update_glm_parameters_mle

    assert len(scope) == 1, "scope of univariate parametric for more than one variable?"
    idx = scope[0]
    # dataOut = data[:, [idx]]

    parametric_type = ds_context.parametric_type[idx]

    assert parametric_type is not None

    node = parametric_type()

    node.scope.append(idx)

    update_glm_parameters_mle(node, data, scope)

    return node
