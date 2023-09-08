import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.sampling import sample
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

    def test_sampling(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

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

        # since RAT-SPNs are completely composed out of tested layers and nodes, the validity does not have to be checked specifically
        # should simply NOT raise an error
        sample(rat, 100)

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

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

        # since RAT-SPNs are completely composed out of tested layers and nodes, the validity does not have to be checked specifically
        # should simply NOT raise an error
        samples_mean = sample(rat, 100).mean()

        for backend in backends:
            tl.set_backend(backend)
            rat_updated = updateBackend(rat)
            rat_samples_updated = sample(rat_updated, 100)
            samples_mean_updated = tl_toNumpy(rat_samples_updated).mean()
            self.assertTrue(np.allclose(samples_mean, samples_mean_updated, atol=0.01, rtol=0.1))


if __name__ == "__main__":
    unittest.main()
