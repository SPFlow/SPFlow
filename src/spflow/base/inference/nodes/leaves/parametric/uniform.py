from spflow.base.structure.nodes.leaves.parametric import (
    Uniform,
    get_scipy_object,
    get_scipy_object_parameters,
)
from multipledispatch import dispatch  # type: ignore

import numpy as np
from typing import Optional


@dispatch(Uniform, data=np.ndarray)  # type: ignore[no-redef]
def node_likelihood(node: Uniform, data: np.ndarray = None) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).pdf(
        x=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(Uniform, data=np.ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: Uniform, data: np.ndarray = None) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    probs[~marg_ids] = get_scipy_object(node).logpdf(
        x=data[~marg_ids], **get_scipy_object_parameters(node)
    )
    return probs
