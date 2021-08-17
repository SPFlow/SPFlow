import unittest
import numpy as np
from spn.python.structure.nodes.node import Node, SumNode, LeafNode, ProductNode, _get_node_counts
from spn.python.structure.nodes.validity_checks import _isvalid_spn


class TestNode(unittest.TestCase):
    def test_spn_fail_scope1(self):
        spn: Node = ProductNode(
            children=[
                LeafNode(scope=[1]),
                LeafNode(scope=[1]),
            ],
            scope=[1, 2],
        )

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_fail_scope2(self):
        spn: Node = ProductNode(
            children=[
                LeafNode(scope=[1]),
                LeafNode(scope=[2]),
            ],
            scope=[1],
        )

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_fail_weights1(self):
        spn: Node = SumNode(
            children=[
                LeafNode(scope=[1]),
                LeafNode(scope=[1]),
            ],
            scope=[1],
            weights=np.array([0.49, 0.49]),
        )

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_fail_weights2(self):
        spn: Node = SumNode(
            children=[
                LeafNode(scope=[1]),
                LeafNode(scope=[1]),
            ],
            scope=[1],
            weights=np.array([1.0]),
        )

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_missing_children(self):
        spn: Node = ProductNode(children=None, scope=[1, 2])

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_fail_leaf_with_children(self):
        spn: Node = SumNode(
            children=[
                LeafNode(scope=[1]),
                LeafNode(scope=[1]),
            ],
            scope=[1],
            weights=np.array([0.5, 0.5]),
        )

        # make sure SPN is valid to begin with
        _isvalid_spn(spn)

        spn.children[0].children.append(LeafNode(scope=[1]))

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_fail_none_children(self):
        spn: Node = ProductNode(
            children=[
                LeafNode(scope=[1]),
                None,
            ],
            scope=[1, 2],
        )

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_tree_small(self):
        spn: Node = ProductNode(
            children=[
                SumNode(
                    children=[
                        LeafNode(scope=[1]),
                        LeafNode(scope=[1]),
                    ],
                    scope=[1],
                    weights=np.array([0.3, 0.7]),
                ),
                LeafNode(scope=[2]),
            ],
            scope=[1, 2],
        )
        _isvalid_spn(spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts([spn])
        self.assertEqual(sum_nodes, 1)
        self.assertEqual(prod_nodes, 1)
        self.assertEqual(leaf_nodes, 3)

    def test_spn_tree_big(self):
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

        _isvalid_spn(spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts([spn])
        self.assertEqual(sum_nodes, 4)
        self.assertEqual(prod_nodes, 5)
        self.assertEqual(leaf_nodes, 11)

    def test_spn_graph_small(self):
        leaf1 = LeafNode(scope=[1])
        leaf2 = LeafNode(scope=[2])
        prod1 = ProductNode(children=[leaf1, leaf2], scope=[1, 2])
        prod2 = ProductNode(children=[leaf1, leaf2], scope=[1, 2])
        sum = SumNode(children=[prod1, prod2], scope=[1, 2], weights=np.array([0.3, 0.7]))

        _isvalid_spn(sum)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts([sum])
        self.assertEqual(sum_nodes, 1)
        self.assertEqual(prod_nodes, 2)
        self.assertEqual(leaf_nodes, 2)

    def test_spn_graph_medium(self):
        leaf_11 = LeafNode(scope=[1])
        leaf_12 = LeafNode(scope=[1])
        leaf_21 = LeafNode(scope=[2])
        leaf_22 = LeafNode(scope=[2])
        sum_11 = SumNode(children=[leaf_11, leaf_12], scope=[1], weights=np.array([0.3, 0.7]))
        sum_12 = SumNode(children=[leaf_11, leaf_12], scope=[1], weights=np.array([0.9, 0.1]))
        sum_21 = SumNode(children=[leaf_21, leaf_22], scope=[2], weights=np.array([0.4, 0.6]))
        sum_22 = SumNode(children=[leaf_21, leaf_22], scope=[2], weights=np.array([0.8, 0.2]))
        prod_11 = ProductNode(children=[sum_11, sum_21], scope=[1, 2])
        prod_12 = ProductNode(children=[sum_11, sum_22], scope=[1, 2])
        prod_13 = ProductNode(children=[sum_12, sum_21], scope=[1, 2])
        prod_14 = ProductNode(children=[sum_12, sum_22], scope=[1, 2])
        sum = SumNode(
            children=[prod_11, prod_12, prod_13, prod_14],
            scope=[1, 2],
            weights=np.array([0.1, 0.2, 0.3, 0.4]),
        )

        _isvalid_spn(sum)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts([sum])
        self.assertEqual(sum_nodes, 5)
        self.assertEqual(prod_nodes, 4)
        self.assertEqual(leaf_nodes, 4)


if __name__ == "__main__":
    unittest.main()
