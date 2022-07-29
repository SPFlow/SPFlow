import unittest
from spflow.base.structure.nodes.validity_checks import _isvalid_spn
from spflow.base.structure.rat.rat_spn import construct_spn, RatSpn
from spflow.base.structure.rat.region_graph import random_region_graph
from spflow.base.structure.nodes.node import _get_node_counts
from spflow.base.inference.module import likelihood
import numpy as np
from spflow.base.learning.context import RandomVariableContext  # type: ignore
from spflow.base.structure.nodes.leaves.parametric import (
    Gaussian,
    get_scipy_object,
    get_scipy_object_parameters,
)


class TestRatSpn(unittest.TestCase):
    def test_rat_spn_num_nodes(self):
        random_variables = set(range(0, 7))
        depth = 2
        replicas = 1
        region_graph = random_region_graph(random_variables, depth, replicas)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * len(region_graph.root_region.random_variables)
        )

        num_nodes_root = 0
        num_nodes_region = 1
        num_nodes_leaf = 1
        with self.assertRaises(ValueError):
            construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf, context)

        num_nodes_root = 1
        num_nodes_region = 0
        with self.assertRaises(ValueError):
            construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf, context)

        num_nodes_region = 1
        num_nodes_leaf = 0
        with self.assertRaises(ValueError):
            construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf, context)

    def test_rat_spn_1(self):
        random_variables = set(range(0, 7))
        depth = 2
        replicas = 1
        region_graph = random_region_graph(random_variables, depth, replicas)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * len(region_graph.root_region.random_variables)
        )

        num_nodes_root = 1
        num_nodes_region = 1
        num_nodes_leaf = 1
        rat_spn = construct_spn(
            region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf, context
        )[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 4)
        self.assertEqual(prod_nodes, 6)
        self.assertEqual(leaf_nodes, 7)

    def test_rat_spn_2(self):
        random_variables = set(range(0, 7))
        depth = 3
        replicas = 1
        region_graph = random_region_graph(random_variables, depth, replicas)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * len(region_graph.root_region.random_variables)
        )

        num_nodes_root = 1
        num_nodes_region = 1
        num_nodes_leaf = 1
        rat_spn = construct_spn(
            region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf, context
        )[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 7)
        self.assertEqual(prod_nodes, 6)
        self.assertEqual(leaf_nodes, 7)

    def test_rat_spn_3(self):
        random_variables = set(range(0, 7))
        depth = 3
        replicas = 2
        region_graph = random_region_graph(random_variables, depth, replicas)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * len(region_graph.root_region.random_variables)
        )

        num_nodes_root = 2
        num_nodes_region = 2
        num_nodes_leaf = 2
        rat_spn = construct_spn(
            region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf, context
        )[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 23)
        self.assertEqual(prod_nodes, 48)
        self.assertEqual(leaf_nodes, 28)

    def test_rat_spn_4(self):
        random_variables = set(range(0, 7))
        depth = 3
        replicas = 3
        region_graph = random_region_graph(random_variables, depth, replicas)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * len(region_graph.root_region.random_variables)
        )

        num_nodes_root = 3
        num_nodes_region = 3
        num_nodes_leaf = 3
        rat_spn = construct_spn(
            region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf, context
        )[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 49)
        self.assertEqual(prod_nodes, 162)
        self.assertEqual(leaf_nodes, 63)

    def test_rat_spn_5(self):
        random_variables = set(range(0, 7))
        depth = 2
        replicas = 1
        num_splits = 3
        region_graph = random_region_graph(random_variables, depth, replicas, num_splits)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * len(region_graph.root_region.random_variables)
        )

        num_nodes_root = 1
        num_nodes_region = 1
        num_nodes_leaf = 1
        rat_spn = construct_spn(
            region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf, context
        )[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)

        self.assertEqual(sum_nodes, 3)
        self.assertEqual(prod_nodes, 4)
        self.assertEqual(leaf_nodes, 7)

    def test_rat_spn_6(self):
        random_variables = set(range(0, 9))
        depth = 3
        replicas = 1
        num_splits = 3
        region_graph = random_region_graph(random_variables, depth, replicas, num_splits)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * len(region_graph.root_region.random_variables)
        )

        num_nodes_root = 1
        num_nodes_region = 1
        num_nodes_leaf = 1
        rat_spn = construct_spn(
            region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf, context
        )[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 5)
        self.assertEqual(prod_nodes, 4)
        self.assertEqual(leaf_nodes, 9)

    def test_rat_spn_7(self):
        random_variables = set(range(0, 7))
        depth = 2
        replicas = 2
        num_splits = 3
        region_graph = random_region_graph(random_variables, depth, replicas, num_splits)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * len(region_graph.root_region.random_variables)
        )

        num_nodes_root = 2
        num_nodes_region = 2
        num_nodes_leaf = 2
        rat_spn = construct_spn(
            region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf, context
        )[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 7)
        self.assertEqual(prod_nodes, 40)
        self.assertEqual(leaf_nodes, 28)

    def test_rat_spn_8(self):
        random_variables = set(range(0, 20))
        depth = 3
        replicas = 3
        num_splits = 3
        region_graph = random_region_graph(random_variables, depth, replicas, num_splits)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * len(region_graph.root_region.random_variables)
        )

        num_nodes_root = 3
        num_nodes_region = 3
        num_nodes_leaf = 2
        rat_spn = construct_spn(
            region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf, context
        )[0]

        _isvalid_spn(rat_spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(rat_spn)
        self.assertEqual(sum_nodes, 49)
        self.assertEqual(prod_nodes, 267)
        self.assertEqual(leaf_nodes, 120)

    def test_rat_spn_module(self):
        region_graph = random_region_graph(X=set(range(0, 7)), depth=2, replicas=2)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * len(region_graph.root_region.random_variables)
        )
        rat_spn = construct_spn(region_graph, 3, 2, 2, context)[0]
        rat_spn_module = RatSpn(region_graph, 3, 2, 2, context)
        self.assertTrue(rat_spn_module.output_nodes[0].equals(rat_spn))

    def test_rat_spn_module_inference(self):
        region_graph = random_region_graph(X=set(range(0, 2)), depth=1, replicas=1)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * len(region_graph.root_region.random_variables)
        )
        rat_spn_module = RatSpn(region_graph, 1, 1, 1, context)
        leaf0 = get_scipy_object(rat_spn_module.nodes[0]).logpdf(
            x=[1], **get_scipy_object_parameters(rat_spn_module.nodes[0])
        )
        leaf1 = get_scipy_object(rat_spn_module.nodes[1]).logpdf(
            x=[1], **get_scipy_object_parameters(rat_spn_module.nodes[1])
        )

        self.assertAlmostEqual(
            likelihood(rat_spn_module, np.array([1.0, 1.0]).reshape(-1, 2))[0][0],
            np.exp(leaf0 + leaf1)[0],
        )

    def test_nodes_rat_spn_big(self):
        # create region graph
        rg = random_region_graph(X=set(range(1024)), depth=5, replicas=2, num_splits=2)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * len(rg.root_region.random_variables)
        )

        # create torch rat spn from region graph
        rat = RatSpn(rg, num_nodes_root=4, num_nodes_region=2, num_nodes_leaf=3, context=context)

        # create dummy input data (batch size x random variables)
        dummy_data = np.random.randn(3, 1024).reshape(-1, 1024)

        # compute outputs for node rat spn
        likelihood(rat, dummy_data)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()