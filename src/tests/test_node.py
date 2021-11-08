import unittest
import numpy as np
from spflow.base.structure.nodes.node import (
    INode,
    ISumNode,
    ILeafNode,
    IProductNode,
    _get_node_counts,
)
from spflow.base.structure.nodes.validity_checks import _isvalid_spn


class TestNode(unittest.TestCase):
    def test_spn_fail_scope1(self):
        spn: INode = IProductNode(
            children=[
                ILeafNode(scope=[1]),
                ILeafNode(scope=[1]),
            ],
            scope=[1, 2],
        )

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_fail_scope2(self):
        spn: INode = IProductNode(
            children=[
                ILeafNode(scope=[1]),
                ILeafNode(scope=[2]),
            ],
            scope=[1],
        )

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_fail_weights1(self):
        spn: INode = ISumNode(
            children=[
                ILeafNode(scope=[1]),
                ILeafNode(scope=[1]),
            ],
            scope=[1],
            weights=np.array([0.49, 0.49]),
        )

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_fail_weights2(self):
        spn: INode = ISumNode(
            children=[
                ILeafNode(scope=[1]),
                ILeafNode(scope=[1]),
            ],
            scope=[1],
            weights=np.array([1.0]),
        )

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_missing_children(self):
        spn: INode = IProductNode(children=None, scope=[1, 2])

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_fail_leaf_with_children(self):
        spn: INode = ISumNode(
            children=[
                ILeafNode(scope=[1]),
                ILeafNode(scope=[1]),
            ],
            scope=[1],
            weights=np.array([0.5, 0.5]),
        )

        # make sure SPN is valid to begin with
        _isvalid_spn(spn)

        spn.children[0].children.append(ILeafNode(scope=[1]))

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_fail_none_children(self):
        spn: INode = IProductNode(
            children=[
                ILeafNode(scope=[1]),
                None,
            ],
            scope=[1, 2],
        )

        with self.assertRaises(AssertionError):
            _isvalid_spn(spn)

    def test_spn_tree_small(self):
        spn: INode = IProductNode(
            children=[
                ISumNode(
                    children=[
                        ILeafNode(scope=[1]),
                        ILeafNode(scope=[1]),
                    ],
                    scope=[1],
                    weights=np.array([0.3, 0.7]),
                ),
                ILeafNode(scope=[2]),
            ],
            scope=[1, 2],
        )
        _isvalid_spn(spn)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts([spn])
        self.assertEqual(sum_nodes, 1)
        self.assertEqual(prod_nodes, 1)
        self.assertEqual(leaf_nodes, 3)

    def test_spn_tree_big(self):
        spn: INode = IProductNode(
            children=[
                ISumNode(
                    children=[
                        IProductNode(
                            children=[
                                ISumNode(
                                    children=[
                                        IProductNode(
                                            children=[
                                                ILeafNode(scope=[1]),
                                                ILeafNode(scope=[2]),
                                            ],
                                            scope=[1, 2],
                                        ),
                                        ILeafNode(scope=[1, 2]),
                                    ],
                                    scope=[1, 2],
                                    weights=np.array([0.9, 0.1]),
                                ),
                                ILeafNode(scope=[3]),
                            ],
                            scope=[1, 2, 3],
                        ),
                        IProductNode(
                            children=[
                                ILeafNode(scope=[1]),
                                ISumNode(
                                    children=[
                                        ILeafNode(scope=[2, 3]),
                                        ILeafNode(scope=[2, 3]),
                                    ],
                                    scope=[2, 3],
                                    weights=np.array([0.5, 0.5]),
                                ),
                            ],
                            scope=[1, 2, 3],
                        ),
                        ILeafNode(scope=[1, 2, 3]),
                    ],
                    scope=[1, 2, 3],
                    weights=np.array([0.4, 0.1, 0.5]),
                ),
                ISumNode(
                    children=[
                        IProductNode(
                            children=[
                                ILeafNode(scope=[4]),
                                ILeafNode(scope=[5]),
                            ],
                            scope=[4, 5],
                        ),
                        ILeafNode(scope=[4, 5]),
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
        leaf1 = ILeafNode(scope=[1])
        leaf2 = ILeafNode(scope=[2])
        prod1 = IProductNode(children=[leaf1, leaf2], scope=[1, 2])
        prod2 = IProductNode(children=[leaf1, leaf2], scope=[1, 2])
        sum = ISumNode(children=[prod1, prod2], scope=[1, 2], weights=np.array([0.3, 0.7]))

        _isvalid_spn(sum)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts([sum])
        self.assertEqual(sum_nodes, 1)
        self.assertEqual(prod_nodes, 2)
        self.assertEqual(leaf_nodes, 2)

    def test_spn_graph_medium(self):
        leaf_11 = ILeafNode(scope=[1])
        leaf_12 = ILeafNode(scope=[1])
        leaf_21 = ILeafNode(scope=[2])
        leaf_22 = ILeafNode(scope=[2])
        sum_11 = ISumNode(children=[leaf_11, leaf_12], scope=[1], weights=np.array([0.3, 0.7]))
        sum_12 = ISumNode(children=[leaf_11, leaf_12], scope=[1], weights=np.array([0.9, 0.1]))
        sum_21 = ISumNode(children=[leaf_21, leaf_22], scope=[2], weights=np.array([0.4, 0.6]))
        sum_22 = ISumNode(children=[leaf_21, leaf_22], scope=[2], weights=np.array([0.8, 0.2]))
        prod_11 = IProductNode(children=[sum_11, sum_21], scope=[1, 2])
        prod_12 = IProductNode(children=[sum_11, sum_22], scope=[1, 2])
        prod_13 = IProductNode(children=[sum_12, sum_21], scope=[1, 2])
        prod_14 = IProductNode(children=[sum_12, sum_22], scope=[1, 2])
        sum = ISumNode(
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
