"""
Created on April 15, 2018

@author: Alejandro Molina
"""


from spn.algorithms.Inference import add_node_likelihood, leaf_marginalized_likelihood
from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.parametric.utils import get_scipy_obj_params

POS_EPS = 1e-7


def continuous_likelihood(node, data=None, dtype=np.float64):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype)
    scipy_obj, params = get_scipy_obj_params(node)
    probs[~marg_ids] = scipy_obj.pdf(observations, **params)
    return probs


lognormal_likelihood = continuous_likelihood
exponential_likelihood = continuous_likelihood


def gamma_likelihood(node, data=None, dtype=np.float64):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype)

    observations[observations == 0] += POS_EPS

    scipy_obj, params = get_scipy_obj_params(node)
    probs[~marg_ids] = scipy_obj.pdf(observations, **params)
    return probs


def discrete_likelihood(node, data=None, dtype=np.float64):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype)
    scipy_obj, params = get_scipy_obj_params(node)
    probs[~marg_ids] = scipy_obj.pmf(observations, **params)
    return probs


bernoulli_likelihood = discrete_likelihood
geometric_likelihood = discrete_likelihood


def categorical_likelihood(node, data=None, dtype=np.float64):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype)

    cat_data = observations.astype(np.int64)
    assert np.all(np.equal(np.mod(cat_data, 1), 0))
    out_domain_ids = cat_data >= node.k

    idx_out = ~marg_ids
    idx_out[idx_out] = out_domain_ids
    probs[idx_out] = 0

    idx_in = ~marg_ids
    idx_in[idx_in] = ~out_domain_ids
    probs[idx_in] = np.array(node.p)[cat_data[~out_domain_ids]]
    return probs


def categorical_dictionary_likelihood(node, data=None, dtype=np.float64):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype)

    dict_probs = [node.p.get(val, 0.0) for val in observations]
    probs[~marg_ids] = dict_probs
    return probs


def uniform_likelihood(node, data=None, dtype=np.float64):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype)

    probs[~marg_ids] = node.density
    return probs


def add_parametric_inference_support():
    add_node_likelihood(Gaussian, continuous_likelihood)
    add_node_likelihood(Gamma, gamma_likelihood)
    add_node_likelihood(LogNormal, lognormal_likelihood)
    add_node_likelihood(Poisson, discrete_likelihood)
    add_node_likelihood(Bernoulli, bernoulli_likelihood)
    add_node_likelihood(Categorical, categorical_likelihood)
    add_node_likelihood(Geometric, geometric_likelihood)
    add_node_likelihood(Exponential, exponential_likelihood)
    add_node_likelihood(Uniform, uniform_likelihood)
    add_node_likelihood(CategoricalDictionary, categorical_dictionary_likelihood)
