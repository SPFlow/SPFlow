'''
Created on April 15, 2018

@author: Alejandro Molina
'''
import numpy as np
from scipy.stats import gamma, lognorm

from spn.structure.leaves.conditional.Conditional import Conditional_Gaussian, Conditional_Poisson, \
    Conditional_Bernoulli
import statsmodels.api as sm


def update_glm_parameters_mle(node, data, scope):  # assume data is tuple (output np array, conditional np array)

    assert len(scope) == 1, 'more than one output variable in scope?'
    data = data[~np.isnan(data)].reshape(data.shape)

    dataOut = data[:, :len(scope)]
    dataIn = data[:, len(scope):]

    assert dataOut.shape[1] == 1, 'more than one output variable in scope?'

    if dataOut.shape[0] == 0:
        return

    if isinstance(node, Conditional_Gaussian):
        family = sm.families.Gaussian()
    elif isinstance(node, Conditional_Poisson):
        family = sm.families.Poisson()
    elif isinstance(node, Conditional_Bernoulli):
        family = sm.families.Binomial()
    else:
        raise Exception("Unknown conditional " + str(type(node)))

    dataIn = np.c_[dataIn, np.ones((dataIn.shape[0]))]
    node.weights = sm.GLM(dataOut, dataIn, family=family).fit_regularized(alpha=0.1).params
