from multipledispatch import dispatch  # type: ignore
import numpy as np
from .rat_spn import RatSpn
from spn.base.nodes.inference import log_likelihood


@dispatch(RatSpn, np.ndarray)
def log_likelihood(rat_spn: RatSpn, data: np.ndarray):
    return log_likelihood(rat_spn.root_node, data)


@dispatch(RatSpn, np.ndarray)
def likelihood(rat_spn: RatSpn, data: np.ndarray):
    np.exp(log_likelihood(rat_spn, data))
