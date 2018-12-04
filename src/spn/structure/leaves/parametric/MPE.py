"""
Created on July 02, 2018

@author: Alejandro Molina
"""
from spn.algorithms.MPE import get_mpe_top_down_leaf, add_node_mpe
from spn.structure.leaves.parametric.Inference import (
    continuous_likelihood,
    gamma_likelihood,
    lognormal_likelihood,
    discrete_likelihood,
    bernoulli_likelihood,
    categorical_likelihood,
    geometric_likelihood,
    exponential_likelihood,
    categorical_dictionary_likelihood,
)
from spn.structure.leaves.parametric.Parametric import (
    Gaussian,
    Gamma,
    LogNormal,
    Poisson,
    Bernoulli,
    Categorical,
    Geometric,
    Exponential,
    CategoricalDictionary,
    NegativeBinomial,
    Hypergeometric,
)
import numpy as np


def get_parametric_bottom_up_ll(ll_func, mode_func):
    def param_bu_fn(node, data=None, dtype=np.float64):
        probs = ll_func(node, data=data, dtype=dtype)

        mpe_ids = np.isnan(data[:, node.scope[0]])
        mode_data = np.ones((1, data.shape[1])) * mode_func(node)
        probs[mpe_ids] = ll_func(node, data=mode_data, dtype=dtype)

        return probs

    return param_bu_fn


def get_parametric_top_down_ll(mode_func):
    def param_td_fn(node, input_vals, data=None, lls_per_node=None):
        get_mpe_top_down_leaf(node, input_vals, data=data, mode=mode_func(node))

    return param_td_fn


def add_parametric_mpe_support():
    def gaussian_mode(node):
        return node.mean

    add_node_mpe(
        Gaussian,
        get_parametric_bottom_up_ll(continuous_likelihood, gaussian_mode),
        get_parametric_top_down_ll(gaussian_mode),
    )

    def gamma_mode(node):
        return (node.alpha - 1) / node.beta

    add_node_mpe(
        Gamma, get_parametric_bottom_up_ll(gamma_likelihood, gamma_mode), get_parametric_top_down_ll(gamma_mode)
    )

    def lognormal_mode(node):
        return np.exp(node.mean - node.variance)

    add_node_mpe(
        LogNormal,
        get_parametric_bottom_up_ll(lognormal_likelihood, lognormal_mode),
        get_parametric_top_down_ll(lognormal_mode),
    )

    def poisson_mode(node):
        return np.floor(node.mean)

    add_node_mpe(
        Poisson,
        get_parametric_bottom_up_ll(discrete_likelihood, poisson_mode),
        get_parametric_top_down_ll(poisson_mode),
    )

    def bernoulli_mode(node):
        if node.p > 0.5:
            return 1
        else:
            return 0

    add_node_mpe(
        Bernoulli,
        get_parametric_bottom_up_ll(bernoulli_likelihood, bernoulli_mode),
        get_parametric_top_down_ll(bernoulli_mode),
    )

    def categorical_mode(node):
        return np.argmax(node.p)

    add_node_mpe(
        Categorical,
        get_parametric_bottom_up_ll(categorical_likelihood, categorical_mode),
        get_parametric_top_down_ll(categorical_mode),
    )

    def geometric_mode(node):
        return 1

    add_node_mpe(
        Geometric,
        get_parametric_bottom_up_ll(geometric_likelihood, geometric_mode),
        get_parametric_top_down_ll(geometric_mode),
    )

    def negative_binomial_mode(node):
        if node.n <= 1:
            return 0
        else:
            return np.floor(node.p * (node.n - 1) / (1 - node.p))

    add_node_mpe(
        NegativeBinomial,
        get_parametric_bottom_up_ll(geometric_likelihood, negative_binomial_mode),
        get_parametric_top_down_ll(negative_binomial_mode),
    )

    def exponential_mode(node):
        return 0

    add_node_mpe(
        Exponential,
        get_parametric_bottom_up_ll(exponential_likelihood, exponential_mode),
        get_parametric_top_down_ll(exponential_mode),
    )

    def hypergeometric_mode(node):
        return np.floor((node.n + 1) * (node.K + 1 / (node.N + 2)))

    add_node_mpe(
        Hypergeometric,
        get_parametric_bottom_up_ll(exponential_likelihood, hypergeometric_mode),
        get_parametric_top_down_ll(hypergeometric_mode),
    )

    def categoricaldict_mode(node):
        return node.params.keys()[np.argmax(node.params.values())]

    add_node_mpe(
        CategoricalDictionary,
        get_parametric_bottom_up_ll(categorical_dictionary_likelihood, categoricaldict_mode),
        get_parametric_top_down_ll(categoricaldict_mode),
    )
