"""
Created on August 09, 2021

@authors: Kevin Huy Nguyen

This file provides the sampling methods for parametric leaves.
"""
from spn.python.structure.nodes.node import ILeafNode, Node
from multipledispatch import dispatch  # type: ignore
from spn.python.structure.nodes.leaves.parametric.parametric import (
    ParametricLeaf,
    Gaussian,
    Gamma,
    Poisson,
    LogNormal,
    Geometric,
    Exponential,
    Bernoulli,
    get_scipy_object_parameters,
    get_scipy_object,
)

# TODO: Categorical, CategoricalDictionary

import numpy as np


@dispatch(Node)  # type: ignore[no-redef]
def sample_parametric_node(node: Node, n_samples, rand_gen) -> None:
    """Sample from the associated scipy object of a parametric leaf node.

    The standard implementation accepts nodes of any type and raises an error, if there is no sampling
    procedure implemented for the given node or if the number of wanted samples is not bigger than zero.

    Arguments:
        node:
            The node which is to be sampled.
        n_samples:
            Number of samples to be generated per node.
        rand_gen:
            Seed for random number generator.


    Returns:
        A scipy object representing the distribution of the given node, or None.

    Raises:
        NotImplementedError:
            The node is a ILeafNode and does not provide a scipy object or the node is not a ILeafNode
            and cannot provide a scipy object.

    """
    assert n_samples > 0
    assert isinstance(node, ParametricLeaf)

    if type(node) is ILeafNode:
        raise NotImplementedError(f"{node} does not provide a scipy object")
    else:
        raise NotImplementedError(f"{node} cannot provide scipy objects")


@dispatch(Gaussian)  # type: ignore[no-redef]
def sample_parametric_node(
    node: Gaussian, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


@dispatch(Gamma)  # type: ignore[no-redef]
def sample_parametric_node(
    node: Gamma, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


@dispatch(LogNormal)  # type: ignore[no-redef]
def sample_parametric_node(
    node: LogNormal, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


@dispatch(Poisson)  # type: ignore[no-redef]
def sample_parametric_node(
    node: Poisson, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


@dispatch(Geometric)  # type: ignore[no-redef]
def sample_parametric_node(
    node: Geometric, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


@dispatch(Exponential)  # type: ignore[no-redef]
def sample_parametric_node(
    node: Exponential, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


@dispatch(Bernoulli)  # type: ignore[no-redef]
def sample_parametric_node(
    node: Bernoulli, n_samples: int, rand_gen: np.random.RandomState
) -> np.ndarray:
    scipy_obj, params = get_scipy_object(node), get_scipy_object_parameters(node)

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X
