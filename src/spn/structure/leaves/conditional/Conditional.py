'''
Created on March 20, 2018
@author: Alejandro Molina
'''

import numpy as np

from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import Type


class Conditional(Leaf):
    def __init__(self, type, scope=None, weights=None, evidence_size=None):
        Leaf.__init__(self, scope=scope)
        self._type = type
        self.evidence_size = evidence_size
        self.weights = weights

    @property
    def type(self):
        return self._type

    @property
    def params(self):
        return {'weights': self.weights}


class Conditional_Gaussian(Conditional):
    """
    Implements a conditional univariate gaussian distribution with parameters
    """

    type = Type.REAL

    def __init__(self, weights=None, scope=None, evidence_size=None):
        Conditional.__init__(self, type(self).type, scope=scope, weights=weights, evidence_size=evidence_size)


class Conditional_Poisson(Conditional):
    """
    Implements a univariate Poisson distribution with parameter
    """

    type = Type.COUNT

    def __init__(self, weights=None, scope=None, evidence_size=None):
        Conditional.__init__(self, type(self).type, scope=scope, weights=weights, evidence_size=evidence_size)


class Conditional_Bernoulli(Conditional):
    """
    Implements a univariate Bernoulli distribution with parameter
    """

    type = Type.BINARY

    def __init__(self, weights=None, scope=None, evidence_size=None):
        Conditional.__init__(self, type(self).type, scope=scope, weights=weights, evidence_size=evidence_size)


def create_conditional_leaf(data, ds_context, scope):
    from spn.structure.leaves.conditional.MLE import update_glm_parameters_mle

    assert len(scope) == 1, "scope of univariate parametric for more than one variable?"
    idx = scope[0]
    # dataOut = data[:, [idx]]

    contidional_type = ds_context.parametric_types[idx]

    assert contidional_type is not None

    node = contidional_type(scope=idx, evidence_size=data.shape[1] - len(scope))

    update_glm_parameters_mle(node, data, scope)

    return node
