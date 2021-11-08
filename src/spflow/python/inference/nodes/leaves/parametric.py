"""
Created on July 08, 2021

@authors: Kevin Huy Nguyen

This file provides the inference functions for the ILeafNodes.
"""

from spflow.python.structure.nodes.leaves.parametric.parametric import (
    get_scipy_object_parameters,
    get_scipy_object,
    Gaussian,
    LogNormal,
    MultivariateGaussian,
    Uniform,
    Bernoulli,
    Binomial,
    NegativeBinomial,
    Poisson,
    Geometric,
    Hypergeometric,
    Exponential,
    Gamma,
)
from typing import Optional
from multipledispatch import dispatch  # type: ignore
import numpy as np
from numpy import ndarray
from spflow.python.structure.nodes import INode

# TODO:
# Binomial, Negative Binomial?
POS_EPS = np.finfo(float).eps
MIN_NEG = np.finfo(float).min


@dispatch(INode, data=ndarray)  # type: ignore[no-redef]
def node_likelihood(node: INode, data: ndarray) -> None:
    """Calculates the likelihood of node depending on the given data.

    The standard implementation accepts nodes of any type and raises an error, if it is a leaf node that does not have
    a likelihood calculation we support yet.

    Arguments:
        node:
            INode to calculate likelihood value of.
        data:
            Data given to evaluate INode.

    Returns:
        np.array with likelihood value for node.

    Raises:
        NotImplementedError:
            The node is a ILeafNode and does not provide parameters or the node is not a ILeafNode.
    """
    raise NotImplementedError(f"Likelihood not provided for {node}.")


@dispatch(Gaussian, data=ndarray)  # type: ignore[no-redef]
def node_likelihood(node: Gaussian, data) -> np.ndarray:
    probs = np.ones((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).pdf(
        x=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(LogNormal, data=ndarray)  # type: ignore[no-redef]
def node_likelihood(node: LogNormal, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).pdf(
        x=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(MultivariateGaussian, data=ndarray)  # type: ignore[no-redef]
def node_likelihood(node: MultivariateGaussian, data) -> np.ndarray:
    probs = np.ones((data.shape[0], 1))
    # TODO: check for partially marginalization
    probs[:, 0] = get_scipy_object(node).pdf(
        data[:, node.scope], **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(Bernoulli, data=ndarray)  # type: ignore[no-redef]
def node_likelihood(node: Bernoulli, data) -> np.ndarray:
    probs = np.ones((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).pmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    probs[probs == 1.0] = 0.999999999
    probs[np.isinf(probs)] = 0.000000001
    return probs


@dispatch(Binomial, data=ndarray)  # type: ignore[no-redef]
def node_likelihood(node: Binomial, data) -> np.ndarray:
    probs = np.ones((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).pmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    probs[probs == 1.0] = 0.999999999
    probs[np.isinf(probs)] = 0.000000001
    return probs


@dispatch(NegativeBinomial, data=ndarray)  # type: ignore[no-redef]
def node_likelihood(node: NegativeBinomial, data) -> np.ndarray:
    probs = np.ones((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).pmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    probs[probs == 1.0] = 0.999999999
    probs[np.isinf(probs)] = 0.000000001
    return probs


@dispatch(Geometric, data=ndarray)  # type: ignore[no-redef]
def node_likelihood(node: Geometric, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).pmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    probs[probs == 1.0] = 0.999999999
    probs[np.isinf(probs)] = 0.000000001
    return probs


@dispatch(Poisson, data=ndarray)  # type: ignore[no-redef]
def node_likelihood(node: Poisson, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).pmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    probs[probs == 1.0] = 0.999999999
    probs[np.isinf(probs)] = 0.000000001
    return probs


@dispatch(Hypergeometric, data=ndarray)  # type: ignore[no-redef]
def node_likelihood(node: Hypergeometric, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).pmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    probs[probs == 1.0] = 0.999999999
    probs[np.isinf(probs)] = 0.000000001
    return probs


@dispatch(Exponential, data=ndarray)  # type: ignore[no-redef]
def node_likelihood(node: Exponential, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).pdf(
        x=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(Gamma, data=ndarray)  # type: ignore[no-redef]
def node_likelihood(node: Gamma, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    observations = data[~marg_ids]
    observations[observations == 0] += POS_EPS
    probs[~marg_ids] = get_scipy_object(node).pdf(
        x=observations, **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(Uniform, data=ndarray)  # type: ignore[no-redef]
def node_likelihood(node: Uniform, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).pdf(
        x=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(INode, data=ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: INode, data: ndarray = None):
    """Calculates the log-likelihood of node depending on the given data.

    The standard implementation accepts nodes of any type and raises an error, if it is a leaf node that does not have
    a log-ikelihood calculation we support yet.

    Arguments:
        node:
            INode to calculate log-likelihood value of.
        data:
            Data given to evaluate INode.

    Returns:
        np.array with log-likelihood value for node.

    Raises:
        NotImplementedError:
            The node is a ILeafNode and does not provide parameters or the node is not a ILeafNode.
    """
    raise NotImplementedError(f"Log-Likelihood not provided for {node}.")


@dispatch(Gaussian, data=ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: Gaussian, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).logpdf(
        x=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(MultivariateGaussian, data=ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: MultivariateGaussian, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    # TODO: check for partially marginalization
    probs[:, 0] = get_scipy_object(node).logpdf(
        data[:, node.scope], **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(Hypergeometric, data=ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: Hypergeometric, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).logpmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    probs[np.isinf(probs)] = MIN_NEG
    return probs


@dispatch(LogNormal, data=ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: LogNormal, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).logpdf(
        x=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(Gamma, data=ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: Gamma, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    observations = data[~marg_ids]
    observations[observations == 0] += POS_EPS
    probs[~marg_ids] = get_scipy_object(node).logpdf(
        x=observations, **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(Poisson, data=ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: Poisson, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).logpmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    probs[np.isinf(probs)] = MIN_NEG
    return probs


@dispatch(Bernoulli, data=ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: Bernoulli, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).logpmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    probs[np.isinf(probs)] = MIN_NEG
    return probs


@dispatch(Binomial, data=ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: Binomial, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).logpmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    probs[np.isinf(probs)] = MIN_NEG
    return probs


@dispatch(NegativeBinomial, data=ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: NegativeBinomial, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).logpmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    probs[np.isinf(probs)] = MIN_NEG
    return probs


@dispatch(Geometric, data=ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: Geometric, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).logpmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    probs[np.isinf(probs)] = MIN_NEG
    return probs


@dispatch(Exponential, data=ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: Exponential, data=None) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).logpdf(
        x=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(Uniform, data=ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: Uniform, data) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).logpdf(
        x=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    return probs
