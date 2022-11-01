import unittest
import torch
import numpy as np
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.torch.structure.autoleaf import (
    AutoLeaf,
    Bernoulli,
    Binomial,
    Exponential,
    Gamma,
    Gaussian,
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
    BernoulliLayer,
    BinomialLayer,
    ExponentialLayer,
    GammaLayer,
    GaussianLayer,
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
from spflow.torch.structure.spn.layers.layer import (
    SPNSumLayer,
    SPNPartitionLayer,
    SPNHadamardLayer,
    marginalize,
)
from spflow.torch.structure.spn.layers.cond_layer import (
    SPNCondSumLayer,
    marginalize,
)
from spflow.torch.structure.spn.nodes.node import (
    SPNSumNode,
    marginalize,
    toBase,
    toTorch,
)
from spflow.torch.structure.spn.nodes.cond_node import (
    SPNCondSumNode,
    marginalize,
    toBase,
    toTorch,
)
from spflow.torch.structure.spn.rat.rat_spn import (
    RatSPN,
    marginalize,
    toBase,
    toTorch,
)
from spflow.base.structure.spn.rat.region_graph import random_region_graph
from spflow.base.structure.spn.rat.rat_spn import RatSPN as BaseRatSPN
from spflow.base.structure.spn.nodes.node import SPNSumNode as BaseSPNSumNode
from spflow.base.structure.spn.layers.layer import SPNSumLayer as BaseSPNSumLayer
from spflow.base.structure.spn.layers.layer import (
    SPNPartitionLayer as BaseSPNPartitionLayer,
)
from spflow.base.structure.spn.layers.layer import (
    SPNHadamardLayer as BaseSPNHadamardLayer,
)
from spflow.base.structure.layers.leaves.parametric.gaussian import (
    GaussianLayer as BaseGaussianLayer,
)


leaf_node_classes = (
    Bernoulli,
    Binomial,
    Exponential,
    Gamma,
    Gaussian,
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
        if isinstance(layer, (SPNSumLayer, SPNCondSumLayer)):
            n_sum_nodes += layer.n_out
        # partition
        elif isinstance(layer, SPNPartitionLayer):
            n_product_nodes += layer.n_out
        # multivariate leaf region
        elif isinstance(layer, SPNHadamardLayer):
            n_product_nodes += layer.n_out
        # leaf node
        elif isinstance(layer, leaf_node_classes):
            n_leaf_nodes += 1
        # leaf layer
        elif isinstance(layer, leaf_layer_classes):
            n_leaf_nodes += layer.n_out
        else:
            raise TypeError(f"Encountered unknown layer of type {type(layer)}.")

        layers += list(layer.children())

    return n_sum_nodes, n_product_nodes, n_leaf_nodes


class TestRatSpn(unittest.TestCase):
    def test_rat_spn_initialization(self):

        random_variables = list(range(7))
        scope = Scope(random_variables)
        region_graph = random_region_graph(scope, depth=2, replicas=1)
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        self.assertRaises(
            ValueError,
            RatSPN,
            region_graph,
            feature_ctx,
            n_root_nodes=0,
            n_region_nodes=1,
            n_leaf_nodes=1,
        )
        self.assertRaises(
            ValueError,
            RatSPN,
            region_graph,
            feature_ctx,
            n_root_nodes=1,
            n_region_nodes=0,
            n_leaf_nodes=1,
        )
        self.assertRaises(
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

    def test_rat_spn_1(self):

        random_variables = list(range(7))
        scope = Scope(random_variables)
        region_graph = random_region_graph(scope, depth=2, replicas=1)
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        rat_spn = RatSPN(
            region_graph,
            feature_ctx,
            n_root_nodes=1,
            n_region_nodes=1,
            n_leaf_nodes=1,
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 4)
        self.assertEqual(n_product_nodes, 6)
        self.assertEqual(n_leaf_nodes, 7)

    def test_rat_spn_2(self):

        random_variables = list(range(7))
        scope = Scope(random_variables)
        region_graph = random_region_graph(scope, depth=3, replicas=1)
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        rat_spn = RatSPN(
            region_graph,
            feature_ctx,
            n_root_nodes=1,
            n_region_nodes=1,
            n_leaf_nodes=1,
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 7)
        self.assertEqual(n_product_nodes, 6)
        self.assertEqual(n_leaf_nodes, 7)

    def test_rat_spn_3(self):

        random_variables = list(range(7))
        scope = Scope(random_variables)
        region_graph = random_region_graph(scope, depth=3, replicas=2)
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        rat_spn = RatSPN(
            region_graph,
            feature_ctx,
            n_root_nodes=2,
            n_region_nodes=2,
            n_leaf_nodes=2,
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 23)
        self.assertEqual(n_product_nodes, 48)
        self.assertEqual(n_leaf_nodes, 28)

    def test_rat_spn_4(self):

        random_variables = list(range(7))
        scope = Scope(random_variables)
        region_graph = random_region_graph(scope, depth=3, replicas=3)
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        rat_spn = RatSPN(
            region_graph,
            feature_ctx,
            n_root_nodes=3,
            n_region_nodes=3,
            n_leaf_nodes=3,
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 49)
        self.assertEqual(n_product_nodes, 162)
        self.assertEqual(n_leaf_nodes, 63)

    def test_rat_spn_5(self):

        random_variables = list(range(7))
        scope = Scope(random_variables)
        region_graph = random_region_graph(
            scope, depth=2, replicas=1, n_splits=3
        )
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        rat_spn = RatSPN(
            region_graph,
            feature_ctx,
            n_root_nodes=1,
            n_region_nodes=1,
            n_leaf_nodes=1,
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 3)
        self.assertEqual(n_product_nodes, 4)
        self.assertEqual(n_leaf_nodes, 7)

    def test_rat_spn_6(self):

        random_variables = list(range(9))
        scope = Scope(random_variables)
        region_graph = random_region_graph(
            scope, depth=3, replicas=1, n_splits=3
        )
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        rat_spn = RatSPN(
            region_graph,
            feature_ctx,
            n_root_nodes=1,
            n_region_nodes=1,
            n_leaf_nodes=1,
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 5)
        self.assertEqual(n_product_nodes, 4)
        self.assertEqual(n_leaf_nodes, 9)

    def test_rat_spn_7(self):

        random_variables = list(range(7))
        scope = Scope(random_variables)
        region_graph = random_region_graph(
            scope, depth=2, replicas=2, n_splits=3
        )
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        rat_spn = RatSPN(
            region_graph,
            feature_ctx,
            n_root_nodes=2,
            n_region_nodes=2,
            n_leaf_nodes=2,
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 7)
        self.assertEqual(n_product_nodes, 40)
        self.assertEqual(n_leaf_nodes, 28)

    def test_rat_spn_8(self):

        random_variables = list(range(20))
        scope = Scope(random_variables)
        region_graph = random_region_graph(
            scope, depth=3, replicas=3, n_splits=3
        )
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        rat_spn = RatSPN(
            region_graph,
            feature_ctx,
            n_root_nodes=3,
            n_region_nodes=3,
            n_leaf_nodes=2,
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 49)
        self.assertEqual(n_product_nodes, 267)
        self.assertEqual(n_leaf_nodes, 120)

    def test_conditional_rat(self):

        random_variables = list(range(7))
        scope = Scope(random_variables, [7])  # conditional scope
        region_graph = random_region_graph(scope, depth=2, replicas=1)
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        rat_spn = RatSPN(
            region_graph,
            feature_ctx,
            n_root_nodes=1,
            n_region_nodes=1,
            n_leaf_nodes=1,
        )

        self.assertTrue(isinstance(rat_spn.root_node, SPNCondSumNode))
        self.assertTrue(isinstance(rat_spn.root_region, SPNCondSumLayer))

    def test_rat_spn_backend_conversion_1(self):

        # create region graph
        scope = Scope(list(range(128)))
        region_graph = random_region_graph(
            scope, depth=5, replicas=2, n_splits=2
        )
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        # create torch rat spn from region graph
        torch_rat = RatSPN(
            region_graph,
            feature_ctx,
            n_root_nodes=4,
            n_region_nodes=2,
            n_leaf_nodes=3,
        )

        # change some parameters
        modules = [torch_rat.root_node]

        while modules:
            module = modules.pop()

            # modules consisting of product nodes have no parameters
            # modules consisting of sum nodes are already random
            # only need to randomize parameters of leaf layers
            if isinstance(module, GaussianLayer):
                module.set_params(
                    mean=torch.randn(module.mean.shape),
                    std=torch.rand(module.std.shape) + 1e-8,
                )

            modules += list(module.children())

        base_rat = toBase(torch_rat)

        modules = [(torch_rat.root_node, base_rat.root_node)]

        while modules:

            torch_module, base_module = modules.pop()

            if isinstance(torch_module, SPNSumNode):
                if not isinstance(base_module, BaseSPNSumNode):
                    raise TypeError()
                self.assertTrue(
                    torch.allclose(torch_module.weights, torch_module.weights)
                )
            if isinstance(torch_module, SPNSumLayer):
                if not isinstance(base_module, BaseSPNSumLayer):
                    raise TypeError()
                self.assertTrue(
                    torch.allclose(torch_module.weights, torch_module.weights)
                )
            if isinstance(torch_module, SPNPartitionLayer):
                if not isinstance(base_module, BaseSPNPartitionLayer):
                    raise TypeError()
            if isinstance(torch_module, SPNHadamardLayer):
                if not isinstance(base_module, BaseSPNHadamardLayer):
                    raise TypeError()
            if isinstance(torch_module, GaussianLayer):
                if not isinstance(base_module, BaseGaussianLayer):
                    raise TypeError()
                self.assertTrue(
                    torch.allclose(torch_module.mean, torch_module.mean)
                )
                self.assertTrue(
                    torch.allclose(torch_module.std, torch_module.std)
                )

            modules += list(zip(torch_module.children(), base_module.children))

    def test_rat_spn_backend_conversion_2(self):

        # create region graph
        scope = Scope(list(range(128)))
        region_graph = random_region_graph(
            scope, depth=5, replicas=2, n_splits=2
        )
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        # create torch rat spn from region graph
        base_rat = BaseRatSPN(
            region_graph,
            feature_ctx,
            n_root_nodes=4,
            n_region_nodes=2,
            n_leaf_nodes=3,
        )

        # change some parameters
        modules = [base_rat.root_node]

        while modules:
            module = modules.pop()

            # modules consisting of product nodes have no parameters
            # modules consisting of sum nodes are already random
            # only need to randomize parameters of leaf layers
            if isinstance(module, BaseGaussianLayer):
                module.set_params(
                    mean=np.random.randn(*module.mean.shape),
                    std=np.random.rand(*module.std.shape) + 1e-8,
                )

            modules += module.children

        torch_rat = toTorch(base_rat)

        modules = [(torch_rat.root_node, base_rat.root_node)]

        while modules:

            torch_module, base_module = modules.pop()

            if isinstance(torch_module, SPNSumNode):
                if not isinstance(base_module, BaseSPNSumNode):
                    raise TypeError()
                self.assertTrue(
                    torch.allclose(torch_module.weights, torch_module.weights)
                )
            if isinstance(torch_module, SPNSumLayer):
                if not isinstance(base_module, BaseSPNSumLayer):
                    raise TypeError()
                self.assertTrue(
                    torch.allclose(torch_module.weights, torch_module.weights)
                )
            if isinstance(torch_module, SPNPartitionLayer):
                if not isinstance(base_module, BaseSPNPartitionLayer):
                    raise TypeError()
            if isinstance(torch_module, SPNHadamardLayer):
                if not isinstance(base_module, BaseSPNHadamardLayer):
                    raise TypeError()
            if isinstance(torch_module, GaussianLayer):
                if not isinstance(base_module, BaseGaussianLayer):
                    raise TypeError()
                self.assertTrue(
                    torch.allclose(torch_module.mean, torch_module.mean)
                )
                self.assertTrue(
                    torch.allclose(torch_module.std, torch_module.std)
                )

            modules += list(zip(torch_module.children(), base_module.children))


if __name__ == "__main__":
    unittest.main()
