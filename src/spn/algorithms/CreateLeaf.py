'''
Created on March 20, 2018

@author: Alejandro Molina
'''

import numpy as np

from src.spn.structure.leaves.Univariate import Normal, Poisson, Bernoulli


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


def create_leaf_histogram(data, ds_context, scope, alpha):
    assert (len(scope) == 1, "scope of univariate for more than one variable?")
    assert (data.shape[1] == 1, "data has more than one feature?")

    idx = scope[0]

    densities, breaks = compute_histogram_type_wise(data=data_slice.getData(),
                                                                                            feature_type=data_slice.featureType,
                                                                                            domain=data_slice.domain,
                                                                                            alpha=alpha)
