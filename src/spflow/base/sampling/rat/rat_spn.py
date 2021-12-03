"""
Created on August 09, 2021

@authors: Kevin Huy Nguyen

This file provides the sampling methods for Rat-SPNs.
"""

from multipledispatch import dispatch  # type: ignore
import numpy as np
from spflow.base.structure.rat import RatSpn
from spflow.base.sampling.nodes.node import sample_instances


@dispatch(RatSpn, np.ndarray, np.random.RandomState)  # type: ignore[no-redef]
def sample_instances(rat_spn: RatSpn, data: np.ndarray, rand_gen: np.random.RandomState):
    return sample_instances(rat_spn.network_type, rat_spn.output_nodes[0], data, rand_gen)
