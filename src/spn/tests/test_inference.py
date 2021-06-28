import unittest
from spn.base.rat.rat_spn import construct_spn
from spn.base.rat.region_graph import random_region_graph
from spn.base.nodes.node import _get_leaf_nodes
from spn.base.nodes.inference import inference
import numpy as np
from spn.base.nodes.node import (
    Node,
    SumNode,
    LeafNode,
    ProductNode,
)


class TestInference(unittest.TestCase):
    def test_inference_1(self):
        random_variables = set(range(1, 8))
        depth = 2
        replicas = 1
        region_graph = random_region_graph(random_variables, depth, replicas)

        num_nodes_root = 1
        num_nodes_region = 1
        num_nodes_leaf = 1
        rat_spn = construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)
        leaves = _get_leaf_nodes(rat_spn.root_node)
        for leaf in leaves:
            leaf.value = 2
        result = inference(rat_spn.root_node)
        self.assertAlmostEqual(result, 16)

    def test_inference_2(self):
        random_variables = set(range(1, 8))
        depth = 2
        replicas = 2
        region_graph = random_region_graph(random_variables, depth, replicas)

        num_nodes_root = 3
        num_nodes_region = 2
        num_nodes_leaf = 2
        rat_spn = construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)
        leaves = _get_leaf_nodes(rat_spn.root_node)
        for leaf in leaves:
            leaf.value = 2
        result = inference(rat_spn.root_node)
        self.assertAlmostEqual(result, 16)

    def test_inference_3(self):
        random_variables = set(range(1, 8))
        depth = 1
        replicas = 2
        region_graph = random_region_graph(random_variables, depth, replicas)

        num_nodes_root = 3
        num_nodes_region = 2
        num_nodes_leaf = 2
        rat_spn = construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)
        leaves = _get_leaf_nodes(rat_spn.root_node)
        for leaf in leaves:
            leaf.value = 2
        result = inference(rat_spn.root_node)
        self.assertAlmostEqual(result, 4)

    def test_inference_4(self):
        random_variables = set(range(1, 8))
        depth = 2
        replicas = 2
        region_graph = random_region_graph(random_variables, depth, replicas)

        num_nodes_root = 3
        num_nodes_region = 2
        num_nodes_leaf = 2
        rat_spn = construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)
        leaves = _get_leaf_nodes(rat_spn.root_node)
        for leaf in leaves:
            leaf.value = 0.5
        result = inference(rat_spn.root_node)
        self.assertAlmostEqual(result, 0.0625)

    def test_inference_5(self):
        random_variables = set(range(1, 8))
        depth = 2
        replicas = 2
        region_graph = random_region_graph(random_variables, depth, replicas)

        num_nodes_root = 3
        num_nodes_region = 2
        num_nodes_leaf = 2
        rat_spn = construct_spn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf)
        leaves = _get_leaf_nodes(rat_spn.root_node)
        i = 0
        while i + 4 < 17:
            leaves[i : i + 4] = sorted(leaves[i : i + 4], key=lambda child: child.scope[0])
            i += 4
        i = 0
        for leaf in leaves:
            leaf.value = np.floor(i / 2)
            i += 1
        result = inference(rat_spn.root_node)
        self.assertAlmostEqual(result, 420)

    def test_inference_6(self):
        spn: Node = ProductNode(
            children=[
                SumNode(
                    children=[
                        ProductNode(
                            children=[
                                SumNode(
                                    children=[
                                        ProductNode(
                                            children=[
                                                LeafNode(scope=[1]),
                                                LeafNode(scope=[2]),
                                            ],
                                            scope=[1, 2],
                                        ),
                                        LeafNode(scope=[1, 2]),
                                    ],
                                    scope=[1, 2],
                                    weights=np.array([0.9, 0.1]),
                                ),
                                LeafNode(scope=[3]),
                            ],
                            scope=[1, 2, 3],
                        ),
                        ProductNode(
                            children=[
                                LeafNode(scope=[1]),
                                SumNode(
                                    children=[
                                        LeafNode(scope=[2, 3]),
                                        LeafNode(scope=[2, 3]),
                                    ],
                                    scope=[2, 3],
                                    weights=np.array([0.5, 0.5]),
                                ),
                            ],
                            scope=[1, 2, 3],
                        ),
                        LeafNode(scope=[1, 2, 3]),
                    ],
                    scope=[1, 2, 3],
                    weights=np.array([0.4, 0.1, 0.5]),
                ),
                SumNode(
                    children=[
                        ProductNode(
                            children=[
                                LeafNode(scope=[4]),
                                LeafNode(scope=[5]),
                            ],
                            scope=[4, 5],
                        ),
                        LeafNode(scope=[4, 5]),
                    ],
                    scope=[4, 5],
                    weights=np.array([0.5, 0.5]),
                ),
            ],
            scope=[1, 2, 3, 4, 5],
        )
        leaves = _get_leaf_nodes(spn)
        for leaf in leaves:
            leaf.value = 0.5
        result = inference(spn)
        self.assertAlmostEqual(result, 99 / 800)


if __name__ == "__main__":
    unittest.main()
