"""
Created on November 27, 2021

@authors: Kevin Huy Nguyen

"""

from multipledispatch import dispatch  # type: ignore
import numpy as np
from spflow.base.structure.rat import RatSpn
from spflow.base.inference.nodes import log_likelihood, likelihood


@dispatch(RatSpn, np.ndarray)  # type: ignore[no-redef]
def log_likelihood(rat_spn: RatSpn, data: np.ndarray) -> np.ndarray:
    return log_likelihood(rat_spn.output_nodes[0], data, rat_spn.network_type)


@dispatch(RatSpn, np.ndarray)  # type: ignore[no-redef]
def likelihood(rat_spn: RatSpn, data: np.ndarray) -> np.ndarray:
    return likelihood(rat_spn.output_nodes[0], data, rat_spn.network_type)
