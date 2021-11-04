from multipledispatch import dispatch  # type: ignore
import numpy as np
from spflow.base.structure.rat import RatSpn
from spflow.base.inference.nodes import log_likelihood


@dispatch(RatSpn, np.ndarray)  # type: ignore[no-redef]
def log_likelihood(rat_spn: RatSpn, data: np.ndarray):
    return log_likelihood(rat_spn.network_type, rat_spn.root_node, data)


@dispatch(RatSpn, np.ndarray)  # type: ignore[no-redef]
def likelihood(rat_spn: RatSpn, data: np.ndarray):
    return np.exp(log_likelihood(rat_spn.network_type, rat_spn.root_node, data))
