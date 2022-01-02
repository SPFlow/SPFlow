"""
Created on November 06, 2021

@authors: Kevin Huy Nguyen
"""

from spflow.base.structure.nodes.leaves.parametric import (
    MultivariateGaussian,
    get_scipy_object,
    get_scipy_object_parameters,
)
from multipledispatch import dispatch  # type: ignore

import numpy as np


@dispatch(MultivariateGaussian, data=np.ndarray)  # type: ignore[no-redef]
def node_likelihood(node: MultivariateGaussian, data: np.ndarray) -> np.ndarray:

    # initialize probabilities
    probs = np.ones((data.shape[0], 1))

    # select relevant data based on node's scope
    data = data[:, node.scope]

    # create mask based on marginalized instances (NaNs)
    # keeps default value of 1 (0 in log-space)
    marg_ids = np.isnan(data)

    # number of marginalized random variables per instance
    n_marg = marg_ids.sum(axis=-1)

    # in case of partially marginalized instances
    if any((n_marg > 0) & (n_marg < len(node.scope))):
        raise ValueError(f"Partial marginalization not yet supported for MultivariateGaussian.")

    # create masked based on distribution's support
    valid_ids = node.check_support(data[~n_marg.astype(bool)])

    if not valid_ids.all():
        raise ValueError(
            f"Encountered data instances that are not in the support of the MultivariateGaussian distribution."
        )

    # compute probabilities for all non-marginalized instances
    probs[~n_marg.astype(bool), 0] = get_scipy_object(node).pdf(
        x=data[~n_marg.astype(bool)], **get_scipy_object_parameters(node)
    )

    # TODO:
    return probs


@dispatch(MultivariateGaussian, data=np.ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: MultivariateGaussian, data: np.ndarray) -> np.ndarray:

    # initialize probabilities
    probs = np.zeros((data.shape[0], 1))

    # select relevant data based on node's scope
    data = data[:, node.scope]

    # create mask based on marginalized instances (NaNs)
    # keeps default value of 1 (0 in log-space)
    marg_ids = np.isnan(data)

    # number of marginalized random variables per instance
    n_marg = marg_ids.sum(axis=-1)

    # in case of partially marginalized instances
    if any((n_marg > 0) & (n_marg < len(node.scope))):
        raise ValueError(f"Partial marginalization not yet supported for MultivariateGaussian.")

    # create masked based on distribution's support
    valid_ids = node.check_support(data[~n_marg.astype(bool)])

    if not valid_ids.all():
        raise ValueError(
            f"Encountered data instances that are not in the support of the MultivariateGaussian distribution."
        )

    # compute probabilities for all non-marginalized instances
    probs[~n_marg.astype(bool), 0] = get_scipy_object(node).logpdf(
        x=data[~n_marg.astype(bool)], **get_scipy_object_parameters(node)
    )

    # TODO:
    return probs
