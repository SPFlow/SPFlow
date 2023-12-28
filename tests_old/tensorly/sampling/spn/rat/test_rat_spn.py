import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.modules.module import sample
from spflow.structure.spn.rat import RatSPN, random_region_graph
from spflow.structure.spn.rat.rat_spn import updateBackend
from spflow.utils import Tensor
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_sampling(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_dtype(torch.float32)

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


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_dtype(torch.float32)

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
        with tl.backend_context(backend):
            rat_updated = updateBackend(rat)
            rat_samples_updated = sample(rat_updated, 100)
            samples_mean_updated = tle.toNumpy(rat_samples_updated).mean()
            tc.assertTrue(np.allclose(samples_mean, samples_mean_updated, atol=0.1, rtol=0.1))


def test_change_dtype(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_dtype(torch.float32)

    scope = Scope(list(range(128)))
    region_graph = random_region_graph(scope, depth=5, replicas=2, n_splits=2)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    # create torch rat spn from region graph
    layer = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=4,
        n_region_nodes=2,
        n_leaf_nodes=3,
    )
    samples = sample(layer, 100)
    tc.assertTrue(samples.dtype == tl.float32)
    layer.to_dtype(tl.float64)

    samples = sample(layer, 100)
    tc.assertTrue(samples.dtype == tl.float64)


def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    scope = Scope(list(range(128)))
    region_graph = random_region_graph(scope, depth=5, replicas=2, n_splits=2)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    # create torch rat spn from region graph
    layer = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=4,
        n_region_nodes=2,
        n_leaf_nodes=3,
    )
    samples = sample(layer, 100)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return

    tc.assertTrue(samples.device.type == "cpu")
    layer.to_device(cuda)

    samples = sample(layer, 100)
    tc.assertTrue(samples.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
