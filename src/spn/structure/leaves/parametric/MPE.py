"""
Created on July 02, 2018
@author: Alejandro Molina
"""
from scipy.stats import multivariate_normal as mn
from spn.algorithms.MPE import get_mpe_top_down_leaf, add_node_mpe

from spn.structure.leaves.parametric.Inference import continuous_log_likelihood, gamma_log_likelihood, \
    discrete_log_likelihood, categorical_log_likelihood, categorical_dictionary_log_likelihood

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
    MultivariateGaussian
)
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_parametric_bottom_up_log_ll(ll_func, mode_func):
    def param_bu_fn(node, data=None, dtype=np.float64):
        probs = ll_func(node, data=data, dtype=dtype)

        mpe_ids = np.isnan(data[:, node.scope[0]])
        mode_data = np.ones((1, data.shape[1])) * mode_func(node)
        probs[mpe_ids] = ll_func(node, data=mode_data, dtype=dtype)

        return probs

    return param_bu_fn


def get_parametric_top_down_ll(mode_func):
    def param_td_fn(node, input_vals, data=None, lls_per_node=None):
        get_mpe_top_down_leaf(
            node,
            input_vals,
            data=data,
            mode=mode_func(node))

    return param_td_fn


def add_parametric_mpe_support():
    def gaussian_mode(node):
        return node.mean

    add_node_mpe(
        Gaussian,
        get_parametric_bottom_up_log_ll(continuous_log_likelihood, gaussian_mode),
        get_parametric_top_down_ll(gaussian_mode),
    )

    def gamma_mode(node):
        return (node.alpha - 1) / node.beta

    add_node_mpe(
        Gamma, get_parametric_bottom_up_log_ll(gamma_log_likelihood, gamma_mode), get_parametric_top_down_ll(gamma_mode)
    )

    def lognormal_mode(node):
        return np.exp(node.mean - node.variance)

    add_node_mpe(
        LogNormal,
        get_parametric_bottom_up_log_ll(continuous_log_likelihood, lognormal_mode),
        get_parametric_top_down_ll(lognormal_mode),
    )

    def poisson_mode(node):
        return np.floor(node.mean)

    add_node_mpe(
        Poisson,
        get_parametric_bottom_up_log_ll(discrete_log_likelihood, poisson_mode),
        get_parametric_top_down_ll(poisson_mode),
    )

    def bernoulli_mode(node):
        if node.p > 0.5:
            return 1
        else:
            return 0

    add_node_mpe(
        Bernoulli,
        get_parametric_bottom_up_log_ll(discrete_log_likelihood, bernoulli_mode),
        get_parametric_top_down_ll(bernoulli_mode),
    )

    def categorical_mode(node):
        return np.argmax(node.p)

    add_node_mpe(
        Categorical,
        get_parametric_bottom_up_log_ll(categorical_log_likelihood, categorical_mode),
        get_parametric_top_down_ll(categorical_mode),
    )

    def geometric_mode(node):
        return 1

    add_node_mpe(
        Geometric,
        get_parametric_bottom_up_log_ll(discrete_log_likelihood, geometric_mode),
        get_parametric_top_down_ll(geometric_mode),
    )

    def negative_binomial_mode(node):
        if node.n <= 1:
            return 0
        else:
            return np.floor(node.p * (node.n - 1) / (1 - node.p))

    add_node_mpe(
        NegativeBinomial,
        get_parametric_bottom_up_log_ll(discrete_log_likelihood, negative_binomial_mode),
        get_parametric_top_down_ll(negative_binomial_mode),
    )

    def exponential_mode(node):
        return 0

    add_node_mpe(
        Exponential,
        get_parametric_bottom_up_log_ll(continuous_log_likelihood, exponential_mode),
        get_parametric_top_down_ll(exponential_mode),
    )

    def hypergeometric_mode(node):
        return np.floor((node.n + 1) * (node.K + 1 / (node.N + 2)))

    add_node_mpe(
        Hypergeometric,
        get_parametric_bottom_up_log_ll(continuous_log_likelihood, hypergeometric_mode),
        get_parametric_top_down_ll(hypergeometric_mode),
    )

    def categoricaldict_mode(node):
        return node.params.keys()[np.argmax(node.params.values())]

    add_node_mpe(
        CategoricalDictionary,
        get_parametric_bottom_up_log_ll(categorical_dictionary_log_likelihood, categoricaldict_mode),
        get_parametric_top_down_ll(categoricaldict_mode),
    )

##Compute the conditional distribution for a multivariate Gaussian when some entries are nan i.e. unseen##

    def makeconditional(mean, cov):
        def conditionalmodemvg(vec):
            activeset = np.isnan(vec)
            totalnans = np.sum(activeset)

            if(totalnans == 0):
                return mn.pdf(vec, mean, cov)
            if(totalnans == (len(mean))):
                return mn.pdf(mean, mean, cov)
            cov1 = cov[activeset, :]
            cov2 = cov[~activeset, :]
            cov11, cov12 = cov1[:, activeset], cov1[:, ~activeset]
            cov21, cov22 = cov2[:, activeset], cov2[:, ~activeset]

            temp = np.matmul(cov12, np.linalg.inv(cov22))

            schur = cov11 - np.matmul(temp, cov21)

            return 1. / (np.sqrt(2 * 3.14 * np.linalg.det(schur)))
        return conditionalmodemvg

##Infer the conditional mean when some entries are seen##

    def conditionalmean(mean, cov):
        def infercondnl(dvec):
            for i in range(0, len(dvec)):
                activeset = np.isnan(dvec[i])

                totalnans = np.sum(activeset)

                if(totalnans == 0):
                    continue
                if(totalnans == (len(mean))):
                    dvec[i] = mean
                else:
                    cov1 = cov[activeset, :]
                    cov2 = cov[~activeset, :]
                    cov11, cov12 = cov1[:, activeset], cov1[:, ~activeset]
                    cov21, cov22 = cov2[:, activeset], cov2[:, ~activeset]

                    mat = np.matmul(cov12, np.linalg.inv(cov22))
                    arr = dvec[i]
                    arr[activeset] = mean[activeset] + \
                        np.matmul(mat, (arr[~activeset] - mean[~activeset]))

            return dvec
        return infercondnl

    def mvg_bu_ll(node, data, dtype=np.float64):
        probs = np.ones((data.shape[0], 1))
        effdat = data[:, node.scope]
        for i in range(0, len(effdat)):

            lambdacond = makeconditional(
                np.asarray(
                    node.mean), np.asarray(
                    node.sigma))
            probs[i] = lambdacond(effdat[i])

        return probs

    def mvg_td(
            node,
            input_vals,
            data=None,
            lls_per_node=None,
            dtype=np.float64):

        input_vals = input_vals[0]

        if len(input_vals) == 0:
            return None


        temp = data[input_vals, :]

        checksum = np.sum(temp[:, node.scope], axis=-1)

        indices = np.isnan(checksum)


        createcondmean = conditionalmean(
            np.asarray(
                node.mean), np.asarray(
                node.sigma))

        temp = data[input_vals[indices], :]

        temp[:, node.scope] = createcondmean(temp[:, node.scope])

        data[input_vals[indices], :] = temp

        return

    add_node_mpe(MultivariateGaussian, mvg_bu_ll, mvg_td)
