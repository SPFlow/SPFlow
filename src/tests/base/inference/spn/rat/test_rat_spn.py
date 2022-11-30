from spflow.meta.data import Scope, FeatureTypes, FeatureContext
from spflow.base.structure.spn.rat import random_region_graph, RatSPN
from spflow.base.inference import log_likelihood
import numpy as np
import unittest


class TestModule(unittest.TestCase):
    def test_likelihood(self):
        # create region graph
        scope = Scope(list(range(128)))
        region_graph = random_region_graph(
            scope, depth=5, replicas=2, n_splits=2
        )
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        # create torch rat spn from region graph
        rat = RatSPN(
            region_graph,
            feature_ctx,
            n_root_nodes=4,
            n_region_nodes=2,
            n_leaf_nodes=3,
        )

        # create dummy input data (batch size x random variables)
        dummy_data = np.random.randn(3, 128)

        # since RAT-SPNs are completely composed out of tested layers and nodes, the validity does not have to be checked specifically
        # should simply NOT raise an error
        log_likelihood(rat, dummy_data)


if __name__ == "__main__":
    unittest.main()
