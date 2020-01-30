"""
Created on April 15, 2018

@author: Alejandro Molina
"""


from spn.algorithms.Inference import add_node_likelihood, leaf_marginalized_likelihood
from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.parametric.utils import get_scipy_obj_params
import sys
import logging

logger = logging.getLogger(__name__)

POS_EPS = np.finfo(float).eps
MIN_NEG = np.finfo(float).min


def continuous_log_likelihood(node, data=None, dtype=np.float64, **kwargs):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype, log_space=True)
    scipy_obj, params = get_scipy_obj_params(node)
    probs[~marg_ids] = scipy_obj.logpdf(observations, **params)
    return probs

def continuous_likelihood(node, data=None, dtype=np.float64, **kwargs):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype)
    scipy_obj, params = get_scipy_obj_params(node)
    probs[~marg_ids] = scipy_obj.pdf(observations, **params)
    return probs


def continuous_multivariate_likelihood(node, data=None, dtype=np.float64, **kwargs):
    probs = np.ones((data.shape[0], 1), dtype=dtype)
    observations = data[:, node.scope]
    assert not np.any(np.isnan(data))
    scipy_obj, params = get_scipy_obj_params(node)
    probs[:, 0] = scipy_obj.pdf(observations, **params)
    return probs


lognormal_likelihood = continuous_likelihood
exponential_likelihood = continuous_likelihood


def gamma_log_likelihood(node, data=None, dtype=np.float64, **kwargs):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype)

    observations[observations == 0] += POS_EPS

    scipy_obj, params = get_scipy_obj_params(node)
    probs[~marg_ids] = scipy_obj.logpdf(observations, **params)
    return probs


def discrete_log_likelihood(node, data=None, dtype=np.float64, **kwargs):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype, log_space=True)
    scipy_obj, params = get_scipy_obj_params(node)
    probs[~marg_ids] = scipy_obj.logpmf(observations, **params)
    # probs[probs == 1.0] = 0.999999999
    # probs[np.isinf(probs)] = 0.000000001
    probs[np.isinf(probs)] = MIN_NEG  # 0.000000001
    return probs

def discrete_likelihood(node, data=None, dtype=np.float64, **kwargs):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype)
    scipy_obj, params = get_scipy_obj_params(node)
    probs[~marg_ids] = scipy_obj.pmf(observations, **params)
    probs[probs == 1.0] = 0.999999999
    probs[np.isinf(probs)] = 0.000000001
    #probs[np.isinf(probs)] = MIN_NEG  # 0.000000001
    return probs


def categorical_log_likelihood(node, data=None, dtype=np.float64, **kwargs):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype, log_space=True)

    cat_data = observations.astype(np.int64)
    assert np.all(np.equal(np.mod(cat_data, 1), 0))
    out_domain_ids = cat_data >= node.k

    idx_out = ~marg_ids
    idx_out[idx_out] = out_domain_ids
    probs[idx_out] = 0

    idx_in = ~marg_ids
    idx_in[idx_in] = ~out_domain_ids
    probs[idx_in] = np.array(np.log(node.p))[cat_data[~out_domain_ids]]
    return probs


def categorical_dictionary_log_likelihood(node, data=None, dtype=np.float64, **kwargs):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype)

    dict_probs = [np.log(node.p.get(val, 0.0)) for val in observations]
    probs[~marg_ids] = dict_probs
    return probs


def uniform_log_likelihood(node, data=None, dtype=np.float64, **kwargs):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype, log_space=True)

    probs[~marg_ids] = np.log(node.density)
    return probs


def add_parametric_inference_support():
    add_node_likelihood(MultivariateGaussian, lambda_func=continuous_multivariate_likelihood)
    add_node_likelihood(Gaussian, lambda_func=continuous_likelihood, log_lambda_func=continuous_log_likelihood)
    add_node_likelihood(Hypergeometric, log_lambda_func=continuous_log_likelihood)
    add_node_likelihood(Gamma, log_lambda_func=gamma_log_likelihood)
    add_node_likelihood(LogNormal, log_lambda_func=continuous_log_likelihood)
    add_node_likelihood(Poisson, log_lambda_func=discrete_log_likelihood)
    add_node_likelihood(Bernoulli, lambda_func=discrete_likelihood, log_lambda_func=discrete_log_likelihood)
    add_node_likelihood(Categorical, log_lambda_func=categorical_log_likelihood)
    add_node_likelihood(Geometric, log_lambda_func=discrete_log_likelihood)
    add_node_likelihood(Exponential, log_lambda_func=continuous_log_likelihood)
    add_node_likelihood(Uniform, log_lambda_func=uniform_log_likelihood)
    add_node_likelihood(CategoricalDictionary, log_lambda_func=categorical_dictionary_log_likelihood)

