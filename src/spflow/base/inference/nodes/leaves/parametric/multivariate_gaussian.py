from spflow.base.structure.nodes.leaves.parametric import (
    MultivariateGaussian,
    get_scipy_object,
    get_scipy_object_parameters,
)
from multipledispatch import dispatch  # type: ignore

import numpy as np
from typing import Optional


@dispatch(MultivariateGaussian, data=np.ndarray)  # type: ignore[no-redef]
def node_likelihood(node: MultivariateGaussian, data: np.ndarray = None) -> np.ndarray:
    probs = np.ones((data.shape[0], 1))
    # TODO: check for partially marginalization
    probs[:, 0] = get_scipy_object(node).pdf(
        data[:, node.scope], **get_scipy_object_parameters(node)
    )
    return probs


@dispatch(MultivariateGaussian, data=np.ndarray)  # type: ignore[no-redef]
def node_log_likelihood(node: MultivariateGaussian, data: np.ndarray = None) -> np.ndarray:
    probs = np.zeros((data.shape[0], 1))
    # TODO: check for partially marginalization
    probs[:, 0] = get_scipy_object(node).logpdf(
        data[:, node.scope], **get_scipy_object_parameters(node)
    )
    return probs
