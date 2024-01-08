import unittest

import tensorly as tl
import torch

from spflow.modules.rat import random_region_graph
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.structure.spn.rat.rat_spn import updateBackend
from spflow.autoleaf import (
    Bernoulli,
    BernoulliLayer,
    Binomial,
    BinomialLayer,
    CondBernoulli,
    CondBernoulliLayer,
    CondBinomial,
    CondBinomialLayer,
    CondExponential,
    CondExponentialLayer,
    CondGamma,
    CondGammaLayer,
    CondGaussian,
    CondGaussianLayer,
    CondGeometric,
    CondGeometricLayer,
    CondLogNormal,
    CondLogNormalLayer,
    CondMultivariateGaussian,
    CondMultivariateGaussianLayer,
    CondNegativeBinomial,
    CondNegativeBinomialLayer,
    CondPoisson,
    CondPoissonLayer,
    Exponential,
    ExponentialLayer,
    Gamma,
    GammaLayer,
    Geometric,
    GeometricLayer,
    Hypergeometric,
    HypergeometricLayer,
    LogNormal,
    LogNormalLayer,
    MultivariateGaussian,
    MultivariateGaussianLayer,
    NegativeBinomial,
    NegativeBinomialLayer,
    Poisson,
    PoissonLayer,
    Uniform,
    UniformLayer,
)
from spflow.modules.layer import CondSumLayer
from spflow.structure.spn.layer.hadamard_layer import HadamardLayer
from spflow.structure.spn.layer.partition_layer import (
    PartitionLayer,
)
from spflow.structure.spn.layer.sum_layer import SumLayer
from spflow.modules.node import (
    CondSumNode,
    # toBase,
    # toTorch,
)
from spflow.modules.node import (
    SumNode,
    # toBase,
    # toTorch,
)
from spflow.structure.spn.rat.rat_spn import RatSPN  # , toBase, toTorch
from spflow.torch.structure.general.node.leaf.gaussian import Gaussian as TorchGaussian
from spflow.torch.structure.general.layer.leaf.gaussian import GaussianLayer as TorchGaussianLayer
from spflow.base.structure.general.node.leaf.gaussian import Gaussian as Gaussian
from spflow.base.structure.general.layer.leaf.gaussian import GaussianLayer as GaussianLayer

leaf_node_classes = (
    Bernoulli,
    Binomial,
    Exponential,
    Gamma,
    Gaussian,
    TorchGaussian,
    Geometric,
    Hypergeometric,
    LogNormal,
    MultivariateGaussian,
    NegativeBinomial,
    Poisson,
    Uniform,
    CondBernoulli,
    CondBinomial,
    CondExponential,
    CondGamma,
    CondGaussian,
    CondGeometric,
    CondLogNormal,
    CondMultivariateGaussian,
    CondNegativeBinomial,
    CondPoisson,
)

leaf_layer_classes = (
    BernoulliLayer,
    BinomialLayer,
    ExponentialLayer,
    GammaLayer,
    GaussianLayer,
    TorchGaussianLayer,
    GeometricLayer,
    HypergeometricLayer,
    LogNormalLayer,
    MultivariateGaussianLayer,
    NegativeBinomialLayer,
    PoissonLayer,
    UniformLayer,
    CondBernoulliLayer,
    CondBinomialLayer,
    CondExponentialLayer,
    CondGammaLayer,
    CondGaussianLayer,
    CondGeometricLayer,
    CondLogNormalLayer,
    CondMultivariateGaussianLayer,
    CondNegativeBinomialLayer,
    CondPoissonLayer,
)


def get_rat_spn_properties(rat_spn: RatSPN):
    n_sum_nodes = 1  # root node
    n_product_nodes = 0
    n_leaf_nodes = 0

    layers = [rat_spn.root_region]

    while layers:
        layer = layers.pop()

        # internal region
        if isinstance(layer, (SumLayer, CondSumLayer)):
            n_sum_nodes += layer.n_out
            layers += list(layer.children)
        # partition
        elif isinstance(layer, PartitionLayer):
            n_product_nodes += layer.n_out
            layers += list(layer.children)
        # multivariate leaf region
        elif isinstance(layer, HadamardLayer):
            n_product_nodes += layer.n_out
            layers += list(layer.children)
        # leaf node
        elif isinstance(layer, leaf_node_classes):
            n_leaf_nodes += 1
            layers += list(layer.children)
        # leaf layer
        elif isinstance(layer, leaf_layer_classes):
            n_leaf_nodes += layer.n_out
            layers += list(layer.children)
        else:
            raise TypeError(f"Encountered unknown layer of type {type(layer)}.")

        # layers += list(layer.children)

    return n_sum_nodes, n_product_nodes, n_leaf_nodes


tc = unittest.TestCase()


def test_rat_spn_initialization(do_for_all_backends):
    random_variables = list(range(7))
    scope = Scope(random_variables)
    region_graph = random_region_graph(scope, depth=2, replicas=1)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    tc.assertRaises(
        ValueError,
        RatSPN,
        region_graph,
        feature_ctx,
        n_root_nodes=0,
        n_region_nodes=1,
        n_leaf_nodes=1,
    )
    tc.assertRaises(
        ValueError,
        RatSPN,
        region_graph,
        feature_ctx,
        n_root_nodes=1,
        n_region_nodes=0,
        n_leaf_nodes=1,
    )
    tc.assertRaises(
        ValueError,
        RatSPN,
        region_graph,
        feature_ctx,
        n_root_nodes=1,
        n_region_nodes=1,
        n_leaf_nodes=0,
    )

    RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=1,
        n_region_nodes=1,
        n_leaf_nodes=1,
    )


def test_rat_spn_1(do_for_all_backends):
    random_variables = list(range(7))
    scope = Scope(random_variables)
    region_graph = random_region_graph(scope, depth=2, replicas=1)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    rat_spn = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=1,
        n_region_nodes=1,
        n_leaf_nodes=1,
    )

    n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(rat_spn)
    tc.assertEqual(n_sum_nodes, 4)
    tc.assertEqual(n_product_nodes, 6)
    tc.assertEqual(n_leaf_nodes, 7)


def test_rat_spn_2(do_for_all_backends):
    random_variables = list(range(7))
    scope = Scope(random_variables)
    region_graph = random_region_graph(scope, depth=3, replicas=1)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    rat_spn = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=1,
        n_region_nodes=1,
        n_leaf_nodes=1,
    )

    n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(rat_spn)
    tc.assertEqual(n_sum_nodes, 7)
    tc.assertEqual(n_product_nodes, 6)
    tc.assertEqual(n_leaf_nodes, 7)


def test_rat_spn_3(do_for_all_backends):
    random_variables = list(range(7))
    scope = Scope(random_variables)
    region_graph = random_region_graph(scope, depth=3, replicas=2)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    rat_spn = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=2,
        n_region_nodes=2,
        n_leaf_nodes=2,
    )

    n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(rat_spn)
    tc.assertEqual(n_sum_nodes, 23)
    tc.assertEqual(n_product_nodes, 48)
    tc.assertEqual(n_leaf_nodes, 28)


def test_rat_spn_4(do_for_all_backends):
    random_variables = list(range(7))
    scope = Scope(random_variables)
    region_graph = random_region_graph(scope, depth=3, replicas=3)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    rat_spn = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=3,
        n_region_nodes=3,
        n_leaf_nodes=3,
    )

    n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(rat_spn)
    tc.assertEqual(n_sum_nodes, 49)
    tc.assertEqual(n_product_nodes, 162)
    tc.assertEqual(n_leaf_nodes, 63)


def test_rat_spn_5(do_for_all_backends):
    random_variables = list(range(7))
    scope = Scope(random_variables)
    region_graph = random_region_graph(scope, depth=2, replicas=1, n_splits=3)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    rat_spn = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=1,
        n_region_nodes=1,
        n_leaf_nodes=1,
    )

    n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(rat_spn)
    tc.assertEqual(n_sum_nodes, 3)
    tc.assertEqual(n_product_nodes, 4)
    tc.assertEqual(n_leaf_nodes, 7)


def test_rat_spn_6(do_for_all_backends):
    random_variables = list(range(9))
    scope = Scope(random_variables)
    region_graph = random_region_graph(scope, depth=3, replicas=1, n_splits=3)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    rat_spn = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=1,
        n_region_nodes=1,
        n_leaf_nodes=1,
    )

    n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(rat_spn)
    tc.assertEqual(n_sum_nodes, 5)
    tc.assertEqual(n_product_nodes, 4)
    tc.assertEqual(n_leaf_nodes, 9)


def test_rat_spn_7(do_for_all_backends):
    random_variables = list(range(7))
    scope = Scope(random_variables)
    region_graph = random_region_graph(scope, depth=2, replicas=2, n_splits=3)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    rat_spn = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=2,
        n_region_nodes=2,
        n_leaf_nodes=2,
    )

    n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(rat_spn)
    tc.assertEqual(n_sum_nodes, 7)
    tc.assertEqual(n_product_nodes, 40)
    tc.assertEqual(n_leaf_nodes, 28)


def test_rat_spn_8(do_for_all_backends):
    random_variables = list(range(20))
    scope = Scope(random_variables)
    region_graph = random_region_graph(scope, depth=3, replicas=3, n_splits=3)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    rat_spn = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=3,
        n_region_nodes=3,
        n_leaf_nodes=2,
    )

    n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(rat_spn)
    tc.assertEqual(n_sum_nodes, 49)
    tc.assertEqual(n_product_nodes, 267)
    tc.assertEqual(n_leaf_nodes, 120)


def test_conditional_rat(do_for_all_backends):
    random_variables = list(range(7))
    scope = Scope(random_variables, [7])  # conditional scope
    region_graph = random_region_graph(scope, depth=2, replicas=1)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    rat_spn = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=1,
        n_region_nodes=1,
        n_leaf_nodes=1,
    )

    tc.assertTrue(isinstance(rat_spn.root_node, CondSumNode))
    tc.assertTrue(isinstance(rat_spn.root_region, CondSumLayer))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    random_variables = list(range(7))
    scope = Scope(random_variables)
    region_graph = random_region_graph(scope, depth=2, replicas=1)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    rat_spn = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=1,
        n_region_nodes=1,
        n_leaf_nodes=1,
    )

    n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(rat_spn)
    for backend in backends:
        with tl.backend_context(backend):
            rat_spn_updated = updateBackend(rat_spn)
            n_sum_nodes_up, n_product_nodes_up, n_leaf_nodes_up = get_rat_spn_properties(rat_spn_updated)
            tc.assertEqual(n_sum_nodes, n_sum_nodes_up)
            tc.assertEqual(n_product_nodes, n_product_nodes_up)
            tc.assertEqual(n_leaf_nodes, n_leaf_nodes_up)


def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    random_variables = list(range(7))
    scope = Scope(random_variables)
    region_graph = random_region_graph(scope, depth=3, replicas=1)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    rat_spn = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=1,
        n_region_nodes=1,
        n_leaf_nodes=1,
    )
    tc.assertTrue(rat_spn.dtype == tl.float32)
    for m in rat_spn.root_node.modules():
        tc.assertTrue(m.dtype == tl.float32)
        if isinstance(m, SumNode) or isinstance(m, SumLayer):
            tc.assertTrue(m.weights.dtype == tl.float32)

    rat_spn.to_dtype(tl.float64)
    tc.assertTrue(rat_spn.dtype == tl.float64)
    for m in rat_spn.root_node.modules():
        tc.assertTrue(m.dtype == tl.float64)
        if isinstance(m, SumNode) or isinstance(m, SumLayer):
            tc.assertTrue(m.weights.dtype == tl.float64)


def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    torch.set_default_dtype(torch.float32)
    # create model on cpu
    random_variables = list(range(7))
    scope = Scope(random_variables)
    region_graph = random_region_graph(scope, depth=3, replicas=1)
    feature_ctx = FeatureContext(scope, {rv: FeatureTypes.Gaussian for rv in scope.query})

    rat_spn = RatSPN(
        region_graph,
        feature_ctx,
        n_root_nodes=1,
        n_region_nodes=1,
        n_leaf_nodes=1,
    )

    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, rat_spn.to_device, cuda)
        return

    tc.assertTrue(rat_spn.device.type == "cpu")
    for m in rat_spn.root_node.modules():
        tc.assertTrue(m.device.type == "cpu")
        if isinstance(m, SumNode) or isinstance(m, SumLayer):
            tc.assertTrue(m.weights.device.type == "cpu")

    rat_spn.to_device(cuda)
    tc.assertTrue(rat_spn.device.type == "cuda")
    for m in rat_spn.root_node.modules():
        tc.assertTrue(m.device.type == "cuda")
        if isinstance(m, SumNode) or isinstance(m, SumLayer):
            tc.assertTrue(m.weights.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
