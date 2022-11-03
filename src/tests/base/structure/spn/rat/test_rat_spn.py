import unittest
from spflow.meta.data import Scope, FeatureTypes, FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.base.structure.autoleaf import (
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
from spflow.base.structure.spn.nodes.sum_node import SumNode, marginalize
from spflow.base.structure.spn.nodes.cond_sum_node import (
    CondSumNode,
    marginalize,
)
from spflow.base.structure.spn.layers.cond_sum_layer import (
    CondSumLayer,
    marginalize,
)
from spflow.base.structure.spn.layers.sum_layer import (
    SumLayer,
    marginalize,
)
from spflow.base.structure.spn.layers.partition_layer import (
    PartitionLayer,
    marginalize,
)
from spflow.base.structure.spn.layers.hadamard_layer import (
    HadamardLayer,
    marginalize,
)
from spflow.base.structure.spn.rat.rat_spn import RatSPN, marginalize
from spflow.base.structure.spn.rat.region_graph import random_region_graph


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
        if isinstance(layer, (SumLayer, CondSumLayer)):
            n_sum_nodes += layer.n_out
        # partition
        elif isinstance(layer, PartitionLayer):
            n_product_nodes += layer.n_out
        # multivariate leaf region
        elif isinstance(layer, HadamardLayer):
            n_product_nodes += layer.n_out
        # leaf node
        elif isinstance(layer, leaf_node_classes):
            n_leaf_nodes += 1
        # leaf layer
        elif isinstance(layer, leaf_layer_classes):
            n_leaf_nodes += layer.n_out
        else:
            raise TypeError(f"Encountered unknown layer of type {type(layer)}.")

        layers += layer.children

    return n_sum_nodes, n_product_nodes, n_leaf_nodes


class TestRatSpn(unittest.TestCase):
    def test_rat_spn_initialization(self):

        random_variables = list(range(7))
        scope = Scope(random_variables)
        region_graph = random_region_graph(
            Scope(random_variables), depth=2, replicas=1
        )
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

        self.assertTrue(isinstance(rat_spn.root_node, CondSumNode))
        self.assertTrue(isinstance(rat_spn.root_region, CondSumLayer))


if __name__ == "__main__":
    unittest.main()
