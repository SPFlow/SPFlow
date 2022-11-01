from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.structure.rat.region_graph import random_region_graph
from spflow.torch.structure.rat.rat_spn import RatSPN
from spflow.torch.inference.rat.rat_spn import log_likelihood
from spflow.torch.inference.module import log_likelihood
from spflow.torch.inference.nodes.node import log_likelihood
from spflow.torch.inference.nodes.leaves.parametric.gaussian import (
    log_likelihood,
)
from spflow.torch.inference.layers.layer import log_likelihood
from spflow.torch.inference.layers.leaves.parametric.gaussian import (
    log_likelihood,
)
import torch
import numpy as np
import unittest


class TestModule(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_likelihood(self):
        # create region graph
        scope = Scope(list(range(128)))
        region_graph = random_region_graph(
            scope, depth=5, replicas=2, n_splits=2
        )
        feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

        # create torch rat spn from region graph
        rat = RatSPN(
            region_graph, feature_ctx, n_root_nodes=4, n_region_nodes=2, n_leaf_nodes=3
        )

        # create dummy input data (batch size x random variables)
        dummy_data = np.random.randn(3, 128)

        # since RAT-SPNs are completely composed out of tested layers and nodes, the validity does not have to be checked specifically
        # should simply NOT raise an error
        log_likelihood(rat, torch.tensor(dummy_data))


if __name__ == "__main__":
    unittest.main()
