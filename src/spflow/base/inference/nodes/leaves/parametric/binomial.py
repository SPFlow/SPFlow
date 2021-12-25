"""
Created on November 06, 2021

@authors: Kevin Huy Nguyen
"""

from spflow.base.structure.nodes.leaves.parametric import (
    Binomial,
    get_scipy_object,
    get_scipy_object_parameters,
)
from .parametric import MIN_NEG
from multipledispatch import dispatch  # type: ignore

import numpy as np


@dispatch(Binomial, data=np.ndarray)  # type: ignore[no-redef]
def node_likelihood(node: Binomial, data: np.ndarray) -> np.ndarray:

    # initialize probabilities
    probs = np.ones((data.shape[0], 1))

    # select relevant data based on node's scope
    data = data[:, node.scope]

    # create mask based on marginalized instances (NaNs)
    # keeps default value of 1 (0 in log-space)
    marg_ids = np.isnan(data)

    # create masked based on distribution's support
    valid_ids = node.check_support(data[~marg_ids])

    if not all(valid_ids):
        raise ValueError(
            f"Encountered data instances that are not in the support of the Binomial distribution."
        )

    # compute probabilities for all non-marginalized instances
    probs[~marg_ids] = get_scipy_object(node).pmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )

    # TODO:
    probs[probs == 1.0] = 0.999999999
    probs[np.isinf(probs)] = 0.000000001
    return probs


@dispatch(Binomial, data=np.ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: Binomial, data: np.ndarray) -> np.ndarray:

    # initialize probabilities
    probs = np.zeros((data.shape[0], 1))

    # select relevant data based on node's scope
    data = data[:, node.scope]

    # create mask based on marginalized instances (NaNs)
    # keeps default value of 1 (0 in log-space)
    marg_ids = np.isnan(data)

    # create masked based on distribution's support
    valid_ids = node.check_support(data[~marg_ids])

    if not all(valid_ids):
        raise ValueError(
            f"Encountered data instances that are not in the support of the Binomial distribution."
        )

    # compute probabilities for all non-marginalized instances
    probs[~marg_ids] = get_scipy_object(node).logpmf(
        k=data[~marg_ids], **get_scipy_object_parameters(node)
    )

    # TODO:
    probs[np.isinf(probs)] = MIN_NEG

    return probs
