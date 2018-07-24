'''
Created on April 15, 2018

@author: Alejandro Molina
'''
import numpy as np
from scipy.stats import gamma, lognorm

from spn.structure.leaves.conditional.Conditional import Conditional_Gaussian, Conditional_Poisson
import statsmodels.api as sm

def update_glm_parameters_mle(node, data, scope):   # assume data is tuple (output np array, conditional np array)
    assert len(scope) == 1, 'more than one output variable in scope?'
    data = data[~np.isnan(data)]

    idx = scope[0]
    dataOut = data[:, idx]
    dataIn = data[:, ~idx] # todo double check here

    assert dataOut.shape[1] == 1, 'more than one output variable in scope?'

    if dataOut.shape[0] == 0:
        return

    if isinstance(node, Conditional_Gaussian):

        dataOut = np.c_[dataOut, np.ones((dataOut.shape[0]))]
        weights = sm.GLM(dataIn, dataOut, family=sm.families.Gaussian).fit().params
        node.mean = node.inv_linkfunc(np.dot(dataIn, weights))
        # todo node.stdev?

        if np.isclose(node.stdev, 0):
            node.stdev = 0.00000001

    elif isinstance(node, Conditional_Poisson):

        dataOut = np.c_[dataOut, np.ones((dataOut.shape[0]))]
        weights = sm.GLM(dataIn, dataOut, family=sm.families.Poisson()).fit().params
        node.mean = node.inv_linkfunc(np.dot(dataIn, weights))


    else:
        raise Exception("Unknown conditional " + str(type(node)))


if __name__ == '__main__':
    node = Conditional_Gaussian(np.inf, np.inf)
    dataOut = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    dataIn = np.ones((dataOut.shape[0], 5))
    data = np.concatenate((dataOut, dataIn), axis=1)
    scope = [0]
    update_glm_parameters_mle(node, data, scope)
    assert np.isclose(node.mean, np.mean(data))
    assert np.isclose(node.stdev, np.std(data))

    node = Conditional_Poisson(np.inf)
    dataOut = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    dataIn = np.ones((dataOut.shape[0], 5))
    data = np.concatenate((dataOut, dataIn), axis=1)
    scope = [0]
    update_glm_parameters_mle(node, data, scope)
    assert np.isclose(node.mean, np.mean(data))

