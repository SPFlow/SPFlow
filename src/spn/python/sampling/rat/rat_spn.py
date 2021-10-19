from multipledispatch import dispatch  # type: ignore
import numpy as np
from spn.python.structure.rat import RatSpn
from spn.python.sampling.nodes.node import sample_instances


@dispatch(RatSpn, np.ndarray, np.random.RandomState)  # type: ignore[no-redef]
def sample_instances(rat_spn: RatSpn, data: np.ndarray, rand_gen: np.random.RandomState):
    return sample_instances(rat_spn.network_type, rat_spn.root_node, data, rand_gen)
