'''
Created on August 15, 2018

@author: Alejandro Molina
'''
import numpy as np

import pymc3 as pm

from spn.structure.leaves.conditional.Conditional import Conditional_Gaussian, Conditional_Poisson, \
    Conditional_Bernoulli


def update_glm_parameters_bayesian(node, data, scope):  # assume data is tuple (output np array, conditional np array)

    assert len(scope) == 1, 'more than one output variable in scope?'
    data = data[~np.isnan(data)].reshape(data.shape)

    dataOut = data[:, :len(scope)]
    dataIn = data[:, len(scope):]

    assert dataOut.shape[1] == 1, 'more than one output variable in scope?'

    if dataOut.shape[0] == 0:
        return

    if isinstance(node, Conditional_Gaussian):
        family =  pm.glm.families.Normal()
    elif isinstance(node, Conditional_Poisson):
        family =  pm.glm.families.Poisson()
    elif isinstance(node, Conditional_Bernoulli):
        family =  pm.glm.families.Binomial()
    else:
        raise Exception("Unknown conditional " + str(type(node)))

    dataIn = np.c_[dataIn, np.ones((dataIn.shape[0]))]
    bglm = pm.glm.linear.GLM(dataIn, dataOut, intercept=False, family=family)

    print(bglm)
