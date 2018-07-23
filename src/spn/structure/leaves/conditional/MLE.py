'''
Created on April 15, 2018

@author: Alejandro Molina
'''
import numpy as np
from scipy.stats import gamma, lognorm

from spn.structure.leaves.conditional.Conditional import Conditional_Gaussian, Conditional_Poisson
import statsmodels.api as sm

def update_glm_parameters_mle(node, data):   # assume data is tuple (output np array, conditional np array)

    data = data[~np.isnan(data)]

    dataOut, dataIn = data

    assert dataOut.shape[1] == 1

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
    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    update_glm_parameters_mle(node, data)
    assert np.isclose(node.mean, np.mean(data))
    assert np.isclose(node.stdev, np.std(data))

    node = Conditional_Poisson(np.inf)
    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    update_glm_parameters_mle(node, data)
    assert np.isclose(node.mean, np.mean(data))

    node = Conditional_Categorical(np.array([1, 1, 1, 1, 1, 1]) / 6)
    data = np.array([0, 0, 1, 3, 5]).reshape(-1, 1)
    update_glm_parameters_mle(node, data)
    assert np.isclose(node.p[0], 2 / 5)
    assert np.isclose(node.p[1], 1 / 5)
    assert np.isclose(node.p[2], 0)
    assert np.isclose(node.p[3], 1 / 5)
    assert np.isclose(node.p[4], 0)
