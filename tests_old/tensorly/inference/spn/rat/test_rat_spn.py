import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.modules.module import log_likelihood
from spflow.structure.spn.rat import RatSPN, random_region_graph
from spflow.structure.spn.rat.rat_spn import updateBackend
from spflow.utils import Tensor
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_likelihood(do_for_all_backends):
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
    log_likelihood(rat, tl.tensor(dummy_data))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    # scope_list = [int(x) for x in list(range(128))]
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
        with tl.backend_context(backend):
            layer_updated = updateBackend(rat)
            layer_ll_updated = log_likelihood(layer_updated, tl.tensor(dummy_data, dtype=tl.float32))
            tc.assertTrue(np.allclose(tle.toNumpy(ll_result), tle.toNumpy(layer_ll_updated)))


def test_change_dtype(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
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

    ll_result = log_likelihood(rat, tl.tensor(dummy_data, dtype=tl.float32))
    tc.assertTrue(ll_result.dtype == tl.float32)
    rat.to_dtype(tl.float64)
    dummy_data = tl.tensor(dummy_data, dtype=tl.float64)
    layer_ll_up = log_likelihood(rat, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)


def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
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

    ll_result = log_likelihood(rat, tl.tensor(dummy_data, dtype=tl.float32))
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, rat.to_device, cuda)
        return
    tc.assertTrue(ll_result.device.type == "cpu")
    rat.to_device(cuda)
    dummy_data = tl.tensor(dummy_data, device=cuda)
    layer_ll = log_likelihood(rat, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
