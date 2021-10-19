import unittest
from spn.python.structure.nodes.validity_checks import _isvalid_spn
from spn.python.structure.rat.rat_spn import construct_spn, RatSpn
from spn.python.structure.rat.region_graph import random_region_graph
from spn.python.structure.nodes.node import (
    _get_node_counts,
)
from spn.python.inference.rat import likelihood, log_likelihood
import numpy as np


class TestRatSpn(unittest.TestCase):
    def test_rat_spn_num_nodes(self):
        random_variables = set(range(1, 8))
        depth = 2
        replicas = 1
        region_graph = random_region_graph(random_variables, depth, replicas)

        num_nodes_root = 0
        num_nodes_region = 1
        num_nodes_leaf = 1
        with self.assertRaises(ValueError):
            construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)

        num_nodes_root = 1
        num_nodes_region = 0
        with self.assertRaises(ValueError):
            construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)

        num_nodes_region = 1
        num_nodes_leaf = 0
        with self.assertRaises(ValueError):
            construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)

    def test_rat_spn_1(self):
        random_variables = set(range(1, 8))
        depth = 2
        replicas = 1
        region_graph = random_region_graph(random_variables, depth, replicas)

        num_nodes_root = 1
        num_nodes_region = 1
        num_nodes_leaf = 1
        rat_spn = construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 4)
        self.assertEqual(prod_nodes, 3)
        self.assertEqual(leaf_nodes, 4)

    def test_rat_spn_2(self):
        random_variables = set(range(1, 8))
        depth = 3
        replicas = 1
        region_graph = random_region_graph(random_variables, depth, replicas)

        num_nodes_root = 1
        num_nodes_region = 1
        num_nodes_leaf = 1
        rat_spn = construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 7)
        self.assertEqual(prod_nodes, 6)
        self.assertEqual(leaf_nodes, 7)

    def test_rat_spn_3(self):
        random_variables = set(range(1, 8))
        depth = 3
        replicas = 2
        region_graph = random_region_graph(random_variables, depth, replicas)

        num_nodes_root = 2
        num_nodes_region = 2
        num_nodes_leaf = 2
        rat_spn = construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 23)
        self.assertEqual(prod_nodes, 48)
        self.assertEqual(leaf_nodes, 28)

    def test_rat_spn_4(self):
        random_variables = set(range(1, 8))
        depth = 3
        replicas = 3
        region_graph = random_region_graph(random_variables, depth, replicas)

        num_nodes_root = 3
        num_nodes_region = 3
        num_nodes_leaf = 3
        rat_spn = construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 49)
        self.assertEqual(prod_nodes, 162)
        self.assertEqual(leaf_nodes, 63)

    def test_rat_spn_5(self):
        random_variables = set(range(1, 8))
        depth = 2
        replicas = 1
        num_splits = 3
        region_graph = random_region_graph(random_variables, depth, replicas, num_splits)

        num_nodes_root = 1
        num_nodes_region = 1
        num_nodes_leaf = 1
        rat_spn = construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)

        self.assertEqual(sum_nodes, 3)
        self.assertEqual(prod_nodes, 2)
        self.assertEqual(leaf_nodes, 5)

    def test_rat_spn_6(self):
        random_variables = set(range(1, 10))
        depth = 3
        replicas = 1
        num_splits = 3
        region_graph = random_region_graph(random_variables, depth, replicas, num_splits)

        num_nodes_root = 1
        num_nodes_region = 1
        num_nodes_leaf = 1
        rat_spn = construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 5)
        self.assertEqual(prod_nodes, 4)
        self.assertEqual(leaf_nodes, 9)

    def test_rat_spn_7(self):
        random_variables = set(range(1, 8))
        depth = 2
        replicas = 2
        num_splits = 3
        region_graph = random_region_graph(random_variables, depth, replicas, num_splits)

        num_nodes_root = 2
        num_nodes_region = 2
        num_nodes_leaf = 2
        rat_spn = construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 7)
        self.assertEqual(prod_nodes, 32)
        self.assertEqual(leaf_nodes, 20)

    def test_rat_spn_8(self):
        random_variables = set(range(1, 21))
        depth = 3
        replicas = 3
        num_splits = 3
        region_graph = random_region_graph(random_variables, depth, replicas, num_splits)

        num_nodes_root = 3
        num_nodes_region = 3
        num_nodes_leaf = 2
        rat_spn = construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 49)
        self.assertEqual(prod_nodes, 225)
        self.assertEqual(leaf_nodes, 78)

    def test_rat_spn_module(self):
        region_graph = random_region_graph(X=set(range(0, 7)), depth=2, replicas=2)
        rat_spn = construct_spn(region_graph, 3, 2, 2)[0]
        rat_spn_module = RatSpn(region_graph, 3, 2, 2)
        self.assertTrue(rat_spn_module.root_node.equals(rat_spn))

    def test_rat_spn_module_inference(self):
        region_graph = random_region_graph(X=set(range(0, 2)), depth=1, replicas=1)
        rat_spn_module = RatSpn(region_graph, 1, 1, 1)
        _isvalid_spn(rat_spn_module.root_node)
        self.assertAlmostEqual(
            likelihood(rat_spn_module, np.array([1.0, 1.0]).reshape(-1, 2))[0][0], 0.05854983
        )


if __name__ == "__main__":
    unittest.main()
