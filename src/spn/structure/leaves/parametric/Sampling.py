"""
Created on April 15, 2018

@author: Alejandro Molina
@author: Antonio Vergari
"""
from spn.algorithms.Sampling import add_node_sampling
from spn.structure.leaves.parametric.Parametric import (
    Parametric,
    Gaussian,
    Gamma,
    Poisson,
    Categorical,
    LogNormal,
    Geometric,
    Exponential,
    Bernoulli,
)

import numpy as np

from spn.structure.leaves.parametric.utils import get_scipy_obj_params


def sample_parametric_node(node, n_samples, data, rand_gen):
    assert isinstance(node, Parametric)
    assert n_samples > 0

    X = None
    if (
        isinstance(node, Gaussian)
        or isinstance(node, Gamma)
        or isinstance(node, LogNormal)
        or isinstance(node, Poisson)
        or isinstance(node, Geometric)
        or isinstance(node, Exponential)
        or isinstance(node, Bernoulli)
    ):

        scipy_obj, params = get_scipy_obj_params(node)

        X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)

    elif isinstance(node, Categorical):
        X = rand_gen.choice(np.arange(node.k), p=node.p, size=n_samples)

    else:
        raise Exception("Node type unknown: " + str(type(node)))

    return X


def add_parametric_sampling_support():
    add_node_sampling(Gaussian, sample_parametric_node)
    add_node_sampling(Gamma, sample_parametric_node)
    add_node_sampling(LogNormal, sample_parametric_node)
    add_node_sampling(Poisson, sample_parametric_node)
    add_node_sampling(Geometric, sample_parametric_node)
    add_node_sampling(Exponential, sample_parametric_node)
    add_node_sampling(Bernoulli, sample_parametric_node)
    add_node_sampling(Categorical, sample_parametric_node)
