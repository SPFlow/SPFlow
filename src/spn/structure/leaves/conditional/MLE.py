'''
Created on April 15, 2018

@author: Alejandro Molina
'''
import numpy as np
from scipy.stats import gamma, lognorm

from spn.structure.leaves.conditional.Conditional import Conditional_Gaussian, Conditional_Poisson, Conditional_Bernoulli
import statsmodels.api as sm

def update_glm_parameters_mle(node, data, scope):   # assume data is tuple (output np array, conditional np array)
    print(scope)
    print(np.shape(data))

    assert len(scope) == 1, 'more than one output variable in scope?'
    data = data[~np.isnan(data)].reshape(data.shape)

    dataOut = data[:, :len(scope)]
    dataIn = data[:, len(scope):]

    # num_instance = data.shape[0]
    #
    # output_mask = np.zeros(data.shape, dtype=bool)   # todo check scope and node.scope again
    # output_mask[:, scope] = True
    #
    # dataOut = data[output_mask].reshape(num_instance, -1)
    # dataIn = data[~output_mask].reshape(num_instance, -1)

    assert dataOut.shape[1] == 1, 'more than one output variable in scope?'

    if dataOut.shape[0] == 0:
        return

    if isinstance(node, Conditional_Gaussian):

        dataIn = np.c_[dataIn, np.ones((dataIn.shape[0]))]
        weights = sm.GLM(dataOut, dataIn, family=sm.families.Gaussian).fit().params
        node.mean = node.inv_linkfunc(np.dot(dataIn, weights))
        # todo node.stdev?

        if np.isclose(node.stdev, 0):
            node.stdev = 0.00000001

    elif isinstance(node, Conditional_Poisson):

        dataIn = np.c_[dataIn, np.ones((dataIn.shape[0]))]
        try:
            weights = sm.GLM(dataOut, dataIn, family=sm.families.Poisson()).fit().params
        except Exception:
            print(dataIn)
            print(np.where(dataOut)==1)
            0/0

        node.mean = node.inv_linkfunc(np.dot(dataIn, weights))

    elif isinstance(node, Conditional_Bernoulli):

        dataIn = np.c_[dataIn, np.ones((dataIn.shape[0]))]
        try:
            weights = sm.GLM(dataOut, dataIn, family=sm.families.Binomial()).fit().params
        except Exception:
            print(dataIn)
            print(np.where(dataOut)==1)
            0/0

        node.p = node.inv_linkfunc(np.dot(dataIn, weights))

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

