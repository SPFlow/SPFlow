"""
Created on November 27, 2021

@authors: Kevin Huy Nguyen

"""

from spflow.base.memoize import memoize
import numpy as np
from typing import Dict
from multipledispatch import dispatch  # type: ignore
from spflow.base.structure.module import Module
from spflow.base.inference.nodes.node_modules import log_likelihood, likelihood
from spflow.base.inference.rat.rat_spn import log_likelihood, likelihood


@dispatch(Module, np.ndarray, cache=dict)
@memoize(Module)
def likelihood(module: Module, data: np.ndarray, cache: Dict = {}) -> np.ndarray:
    return likelihood(module, data, module.network_type, cache=cache)


@dispatch(Module, np.ndarray, cache=dict)
@memoize(Module)
def log_likelihood(module: Module, data: np.ndarray, cache: Dict = {}) -> np.ndarray:
    return log_likelihood(module, data, module.network_type, cache=cache)
