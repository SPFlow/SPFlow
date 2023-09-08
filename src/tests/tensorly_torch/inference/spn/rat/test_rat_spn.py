import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.inference import log_likelihood
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.spn.rat import RatSPN, random_region_graph
from spflow.tensorly.structure.spn.rat.rat_spn import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy


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
        region_graph = random_region_graph(scope, depth=5, replicas=2, n_splits=2)
        feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

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
        log_likelihood(rat, torch.tensor(dummy_data))

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        #scope_list = [int(x) for x in list(range(128))]
        scope = Scope([int(x) for x in list(range(128))])
        region_graph = random_region_graph(scope, depth=5, replicas=2, n_splits=2)
        feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

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
        ll_result = log_likelihood(rat, tl.tensor(dummy_data))

        for backend in backends:
            tl.set_backend(backend)
            layer_updated = updateBackend(rat)
            layer_ll_updated = log_likelihood(layer_updated, tl.tensor(dummy_data, dtype = tl.float64))
            self.assertTrue(np.allclose(tl_toNumpy(ll_result), tl_toNumpy(layer_ll_updated)))


if __name__ == "__main__":
    unittest.main()
