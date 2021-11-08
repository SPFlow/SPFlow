"""
Created on November 06, 2021

@authors: Kevin Huy Nguyen
"""

from spflow.base.structure.nodes.leaves.parametric import (
    Gamma,
    get_scipy_object,
    get_scipy_object_parameters,
)
from .parametric import POS_EPS
from multipledispatch import dispatch  # type: ignore

import numpy as np
from typing import Optional


@dispatch(Gamma, data=np.ndarray)  # type: ignore[no-redef]
def node_likelihood(node: Gamma, data: np.ndarray = None) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    observations = data[~marg_ids]
    observations[observations == 0] += POS_EPS
    probs[~marg_ids] = get_scipy_object(node).pdf(
        x=observations, **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(Gamma, data=np.ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: Gamma, data: np.ndarray = None) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    observations = data[~marg_ids]
    observations[observations == 0] += POS_EPS
    probs[~marg_ids] = get_scipy_object(node).logpdf(
        x=observations, **get_scipy_object_parameters(node)
    )
    return probs
