import unittest
from spflow.meta.data.scope import Scope
from spflow.base.structure.layers.layer import (
    SPNSumLayer,
    SPNPartitionLayer,
    SPNHadamardLayer,
    marginalize,
)
from spflow.base.structure.layers.leaves.parametric.gaussian import (
    GaussianLayer,
    marginalize,
)
from spflow.base.structure.nodes.node import SPNSumNode, marginalize
from spflow.base.structure.rat.rat_spn import RatSPN, marginalize
from spflow.base.structure.rat.region_graph import random_region_graph


def get_rat_spn_properties(rat_spn: RatSPN):

    n_sum_nodes = 1  # root node
    n_product_nodes = 0
    n_leaf_nodes = 0

    layers = [rat_spn.root_region]

    while layers:
        layer = layers.pop()

        # internal region
        if isinstance(layer, SPNSumLayer):
            n_sum_nodes += layer.n_out
        # partition
        elif isinstance(layer, SPNPartitionLayer):
            n_product_nodes += layer.n_out
        # multivariate leaf region
        elif isinstance(layer, SPNHadamardLayer):
            n_product_nodes += layer.n_out
        elif isinstance(layer, GaussianLayer):
            n_leaf_nodes += layer.n_out
        else:
            raise TypeError(f"Encountered unknown layer of type {type(layer)}.")

        layers += layer.children

    return n_sum_nodes, n_product_nodes, n_leaf_nodes


class TestRatSpn(unittest.TestCase):
    def test_rat_spn_initialization(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=2, replicas=1
        )

        self.assertRaises(
            ValueError,
            RatSPN,
            region_graph,
            n_root_nodes=0,
            n_region_nodes=1,
            n_leaf_nodes=1,
        )
        self.assertRaises(
            ValueError,
            RatSPN,
            region_graph,
            n_root_nodes=1,
            n_region_nodes=0,
            n_leaf_nodes=1,
        )
        self.assertRaises(
            ValueError,
            RatSPN,
            region_graph,
            n_root_nodes=1,
            n_region_nodes=1,
            n_leaf_nodes=0,
        )

    def test_rat_spn_1(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=2, replicas=1
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=1, n_region_nodes=1, n_leaf_nodes=1
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 4)
        self.assertEqual(n_product_nodes, 6)
        self.assertEqual(n_leaf_nodes, 7)

    def test_rat_spn_2(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=3, replicas=1
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=1, n_region_nodes=1, n_leaf_nodes=1
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 7)
        self.assertEqual(n_product_nodes, 6)
        self.assertEqual(n_leaf_nodes, 7)

    def test_rat_spn_3(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=3, replicas=2
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=2, n_region_nodes=2, n_leaf_nodes=2
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 23)
        self.assertEqual(n_product_nodes, 48)
        self.assertEqual(n_leaf_nodes, 28)

    def test_rat_spn_4(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=3, replicas=3
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=3, n_region_nodes=3, n_leaf_nodes=3
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 49)
        self.assertEqual(n_product_nodes, 162)
        self.assertEqual(n_leaf_nodes, 63)

    def test_rat_spn_5(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=2, replicas=1, n_splits=3
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=1, n_region_nodes=1, n_leaf_nodes=1
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 3)
        self.assertEqual(n_product_nodes, 4)
        self.assertEqual(n_leaf_nodes, 7)

    def test_rat_spn_6(self):

        random_variables = list(range(9))
        region_graph = random_region_graph(
            Scope(random_variables), depth=3, replicas=1, n_splits=3
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=1, n_region_nodes=1, n_leaf_nodes=1
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 5)
        self.assertEqual(n_product_nodes, 4)
        self.assertEqual(n_leaf_nodes, 9)

    def test_rat_spn_7(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=2, replicas=2, n_splits=3
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=2, n_region_nodes=2, n_leaf_nodes=2
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 7)
        self.assertEqual(n_product_nodes, 40)
        self.assertEqual(n_leaf_nodes, 28)

    def test_rat_spn_8(self):

        random_variables = list(range(20))
        region_graph = random_region_graph(
            Scope(random_variables), depth=3, replicas=3, n_splits=3
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=3, n_region_nodes=3, n_leaf_nodes=2
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 49)
        self.assertEqual(n_product_nodes, 267)
        self.assertEqual(n_leaf_nodes, 120)


if __name__ == "__main__":
    unittest.main()
