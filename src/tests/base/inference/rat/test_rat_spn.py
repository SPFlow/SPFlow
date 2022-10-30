from spflow.meta.data.scope import Scope
from spflow.base.structure.rat.region_graph import random_region_graph
from spflow.base.structure.rat.rat_spn import RatSPN
from spflow.base.inference.rat.rat_spn import log_likelihood
from spflow.base.inference.module import log_likelihood
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.inference.nodes.leaves.parametric.gaussian import (
    log_likelihood,
)
from spflow.base.inference.layers.layer import log_likelihood
from spflow.base.inference.layers.leaves.parametric.gaussian import (
    log_likelihood,
)
import numpy as np
import unittest


class TestModule(unittest.TestCase):
    def test_likelihood(self):
        # create region graph
        region_graph = random_region_graph(
            scope=Scope(list(range(128))), depth=5, replicas=2, n_splits=2
        )

        # create torch rat spn from region graph
        rat = RatSPN(
            region_graph, n_root_nodes=4, n_region_nodes=2, n_leaf_nodes=3
        )

        # create dummy input data (batch size x random variables)
        dummy_data = np.random.randn(3, 128)

        # since RAT-SPNs are completely composed out of tested layers and nodes, the validity does not have to be checked specifically
        # should simply NOT raise an error
        log_likelihood(rat, dummy_data)


if __name__ == "__main__":
    unittest.main()
