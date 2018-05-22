'''
Created on April 15, 2018

@author: Alejandro Molina
'''
import numpy as np
from scipy.stats import gamma, lognorm

from spn.structure.leaves.parametric.Parametric import Gaussian, Gamma, Poisson, Categorical, LogNormal


def update_parametric_parameters_mle(node, data):
    assert data.shape[1] == 1

    if data.shape[0] == 0:
        return

    if isinstance(node, Gaussian):
        node.mean = np.mean(data)
        node.stdev = np.std(data)

    elif isinstance(node, Gamma):
        # x = np.copy(data)
        # x[np.isclose(x, 0)] = 1e-6
        # gamma_params = gamma.fit(x, floc=0)
        # node.alpha = gamma_params[0]
        # node.beta = 1.0 / gamma_params[2]
        alpha, beta = mle_param_fit_gamma(data)
        node.alpha = alpha
        node.beta = beta

    elif isinstance(node, LogNormal):
        lognorm_params = lognorm.fit(data, floc=0)
        node.mean = np.log(lognorm_params[2])
        node.stdev = lognorm_params[0]

    elif isinstance(node, Poisson):
        node.mean = np.mean(data)

    elif isinstance(node, Categorical):
        psum = 0
        for i in range(node.k):
            node.p[i] = np.sum(data == i)
            psum += node.p[i]
        node.p = node.p / psum

    else:
        raise Exception("Unknown parametric " + str(type(node)))


if __name__ == '__main__':
    node = Gaussian(np.inf, np.inf)
    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    parametric_update_parameters_mle(node, data)
    assert np.isclose(node.mean, np.mean(data))
    assert np.isclose(node.stdev, np.std(data))

    node = Gamma(np.inf, np.inf)
    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    parametric_update_parameters_mle(node, data)
    assert np.isclose(node.alpha / node.beta, np.mean(data))

    node = LogNormal(np.inf, np.inf)
    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    parametric_update_parameters_mle(node, data)
    assert np.isclose(node.mean, np.log(data).mean(), atol=0.00001)
    assert np.isclose(node.stdev, np.log(data).std(), atol=0.00001)

    node = Poisson(np.inf)
    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    parametric_update_parameters_mle(node, data)
    assert np.isclose(node.mean, np.mean(data))

    node = Categorical(np.array([1, 1, 1, 1, 1, 1]) / 6)
    data = np.array([0, 0, 1, 3, 5]).reshape(-1, 1)
    parametric_update_parameters_mle(node, data)
    assert np.isclose(node.p[0], 2 / 5)
    assert np.isclose(node.p[1], 1 / 5)
    assert np.isclose(node.p[2], 0)
    assert np.isclose(node.p[3], 1 / 5)
    assert np.isclose(node.p[4], 0)
    assert np.isclose(node.p[3], 1 / 5)
