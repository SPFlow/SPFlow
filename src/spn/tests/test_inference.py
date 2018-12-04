import unittest

import numpy as np

from spn.algorithms.Inference import add_node_likelihood, likelihood, log_likelihood
from spn.structure.Base import Leaf, get_nodes_by_type, assign_ids


def identity_ll(node, data, dtype=np.float64, **kwargs):
    probs = np.zeros((data.shape[0], 1), dtype=dtype)
    probs[:] = np.prod(data[:, node.scope], keepdims=True, axis=1)
    return probs


# The multiplier is just for testing purposes, to check that individual nodes add different contributions
def multiply_ll(node, data, dtype=np.float64, **kwargs):
    probs = np.zeros((data.shape[0], 1), dtype=dtype)
    probs[:] = np.prod(data[:, node.scope], keepdims=True, axis=1) * node.multiplier
    return probs


# The multiplier is just for testing purposes, to check that individual nodes add different contributions
def sum_and_multiplier_ll(node, data, dtype=np.float64, **kwargs):
    probs = np.zeros((data.shape[0], 1), dtype=dtype)
    probs[:] = np.sum(data[:, node.scope], keepdims=True, axis=1) * node.multiplier
    return probs


# The sum is just for testing purposes, to check that individual nodes add different contributions
def sums_ll(node, data, dtype=np.float64, **kwargs):
    probs = np.zeros((data.shape[0], 1), dtype=dtype)
    probs[:] = np.sum(data[:, node.scope], keepdims=True, axis=1)
    return probs


class TestInference(unittest.TestCase):
    def assert_correct(self, spn, data, result):
        l = likelihood(spn, data)
        self.assertEqual(l.shape[0], data.shape[0])
        self.assertEqual(l.shape[1], 1)
        self.assertTrue(np.alltrue(np.isclose(result.reshape(-1, 1), l)))
        self.assertTrue(np.alltrue(np.isclose(np.log(l), log_likelihood(spn, data))))
        self.assertTrue(np.alltrue(np.isclose(np.log(l), log_likelihood(spn, data, debug=True))))
        self.assertTrue(np.alltrue(np.isclose(l, likelihood(spn, data, debug=True))))

    def test_type(self):
        add_node_likelihood(Leaf, identity_ll)

        # test that we get basic computations right
        spn = 0.5 * Leaf(scope=[0, 1]) + 0.5 * (Leaf(scope=0) * Leaf(scope=1))
        data = np.random.rand(10, 4)
        l = likelihood(spn, data, dtype=np.float32)
        self.assertEqual(l.dtype, np.float32)

        l = likelihood(spn, data, dtype=np.float128)
        self.assertEqual(l.dtype, np.float128)

    def test_sum_one_dimension(self):
        add_node_likelihood(Leaf, identity_ll)

        # test that we get basic computations right
        spn = 0.5 * Leaf(scope=0) + 0.5 * Leaf(scope=0)
        data = np.random.rand(10, 1)
        self.assert_correct(spn, data, data)

        spn = 0.1 * Leaf(scope=0) + 0.9 * Leaf(scope=0)
        data = np.random.rand(10, 1)
        self.assert_correct(spn, data, data)

        # test that we can pass whatever dataset, and the scopes are being respected
        # this is important for inner nodes
        spn = 0.1 * Leaf(scope=0) + 0.9 * Leaf(scope=0)
        data = np.random.rand(10, 3)
        r = 0.1 * data[:, 0] + 0.9 * data[:, 0]
        r = r.reshape(-1, 1)
        self.assert_correct(spn, data, r)

        # test that it fails if the weights are not normalized
        spn = 0.1 * Leaf(scope=0) + 0.9 * Leaf(scope=0)
        spn.weights[1] = 0.2
        data = np.random.rand(10, 3)
        with self.assertRaises(AssertionError):
            l = likelihood(spn, data)
        with self.assertRaises(AssertionError):
            log_likelihood(spn, data)

        # test the log space
        spn = 0.1 * Leaf(scope=0) + 0.9 * Leaf(scope=0)
        data = np.random.rand(10, 3)
        r = 0.1 * data[:, 0] + 0.9 * data[:, 0]
        r = r.reshape(-1, 1)
        self.assert_correct(spn, data, r)

    def test_hierarchical_sum_one_dimension(self):
        add_node_likelihood(Leaf, identity_ll)

        # test basic computations in a hierarchy + respecting the scopes
        spn = 0.3 * (0.2 * Leaf(scope=0) + 0.8 * Leaf(scope=0)) + 0.7 * (0.4 * Leaf(scope=0) + 0.6 * Leaf(scope=0))
        data = np.random.rand(10, 3)
        self.assert_correct(spn, data, data[:, 0])

        add_node_likelihood(Leaf, multiply_ll)

        # test that the different nodes contribute differently
        spn = 0.3 * (0.2 * Leaf(scope=0) + 0.8 * Leaf(scope=0)) + 0.7 * (0.4 * Leaf(scope=0) + 0.6 * Leaf(scope=0))
        spn.children[0].children[0].multiplier = 2
        spn.children[0].children[1].multiplier = 3
        spn.children[1].children[0].multiplier = 4
        spn.children[1].children[1].multiplier = 5
        data = np.random.rand(10, 3)
        r = 0.3 * (0.2 * 2 * data[:, 0] + 0.8 * 3 * data[:, 0]) + 0.7 * (0.4 * 4 * data[:, 0] + 0.6 * 5 * data[:, 0])
        self.assert_correct(spn, data, r)

    def test_sum_multiple_dimension(self):
        add_node_likelihood(Leaf, identity_ll)

        # test basic computations in multiple dimensions
        spn = 0.5 * Leaf(scope=[0, 1]) + 0.5 * Leaf(scope=[0, 1])
        data = np.random.rand(10, 2)
        l = likelihood(spn, data)
        self.assert_correct(spn, data, data[:, 0] * data[:, 1])

    def test_hierarchical_sum_multiple_dimension(self):
        add_node_likelihood(Leaf, identity_ll)

        # test basic computations in a hierarchy
        spn = 0.3 * (0.2 * Leaf(scope=[0, 1]) + 0.8 * Leaf(scope=[0, 1])) + 0.7 * (
            0.4 * Leaf(scope=[0, 1]) + 0.6 * Leaf(scope=[0, 1])
        )
        data = np.random.rand(10, 3)
        self.assert_correct(spn, data, data[:, 0] * data[:, 1])

        add_node_likelihood(Leaf, multiply_ll)

        # test different node contributions
        spn = 0.3 * (0.2 * Leaf(scope=[0, 1]) + 0.8 * Leaf(scope=[0, 1])) + 0.7 * (
            0.4 * Leaf(scope=[0, 1]) + 0.6 * Leaf(scope=[0, 1])
        )

        spn.children[0].children[0].multiplier = 2
        spn.children[0].children[1].multiplier = 3
        spn.children[1].children[0].multiplier = 4
        spn.children[1].children[1].multiplier = 5
        data = np.random.rand(10, 3)
        dprod = data[:, 0] * data[:, 1]
        r = 0.3 * (0.2 * 2 * dprod + 0.8 * 3 * dprod) + 0.7 * (0.4 * 4 * dprod + 0.6 * 5 * dprod)
        self.assert_correct(spn, data, r)

    def test_prod_one_dimension(self):
        add_node_likelihood(Leaf, identity_ll)

        # test basic product
        spn = Leaf(scope=0) * Leaf(scope=1)
        data = np.random.rand(10, 2)
        self.assert_correct(spn, data, data[:, 0] * data[:, 1])

        # test respecting the scopes
        spn = Leaf(scope=0) * Leaf(scope=1)
        data = np.random.rand(10, 3)
        self.assert_correct(spn, data, data[:, 0] * data[:, 1])

    def test_prod_multiple_dimension(self):
        add_node_likelihood(Leaf, sums_ll)

        # test basic computations in multiple dimensions
        spn = Leaf(scope=[0, 1]) * Leaf(scope=[2, 3])
        data = np.random.rand(10, 4)
        r = (data[:, 0] + data[:, 1]) * (data[:, 2] + data[:, 3])
        self.assert_correct(spn, data, r)

    def test_hierarchical_prod_multiple_dimension(self):
        add_node_likelihood(Leaf, identity_ll)

        # test basic computations in a hierarchy
        spn = (Leaf(scope=[0, 1]) * Leaf(scope=[2, 3])) * (Leaf(scope=[4, 5]) * Leaf(scope=[6, 7]))
        data = np.random.rand(10, 8)
        self.assert_correct(spn, data, np.prod(data, axis=1))

        add_node_likelihood(Leaf, sums_ll)

        # test different node contributions
        spn = (Leaf(scope=[0, 1]) * Leaf(scope=[2, 3])) * (Leaf(scope=[4, 5]) * Leaf(scope=[6, 7]))
        data = np.random.rand(10, 10)
        dprod = (
            (data[:, 0] + data[:, 1])
            * (data[:, 2] + data[:, 3])
            * (data[:, 4] + data[:, 5])
            * (data[:, 6] + data[:, 7])
        )
        self.assert_correct(spn, data, dprod)

    def test_handmade_multidim(self):
        add_node_likelihood(Leaf, sum_and_multiplier_ll)

        spn = 0.3 * ((0.9 * (leaf(0, 1) * leaf(1, 2)) + 0.1 * (leaf(0, 3) * leaf(1, 4))) * leaf(2, 5)) + 0.7 * (
            0.6 * leaf([0, 1, 2], 6) + 0.4 * leaf([0, 1, 2], 7)
        )
        data = np.random.rand(10, 10)

        r = 0.3 * (
            (0.9 * (data[:, 0] * 2 * data[:, 1]) + 0.1 * (3 * data[:, 0] * 4 * data[:, 1])) * 5 * data[:, 2]
        ) + 0.7 * (0.6 * 6 * (data[:, 0] + data[:, 1] + data[:, 2]) + 0.4 * 7 * (data[:, 0] + data[:, 1] + data[:, 2]))

        self.assert_correct(spn, data, r)

    def test_ll_matrix(self):
        add_node_likelihood(Leaf, sum_and_multiplier_ll)

        node_1_1_1_1 = leaf(2, 1)
        node_1_1_1_2 = leaf(2, 2)
        node_1_1_1 = 0.7 * node_1_1_1_1 + 0.3 * node_1_1_1_2
        node_1_1_2 = leaf([0, 1], 3)
        node_1_1 = node_1_1_1 * node_1_1_2
        node_1_2_1_1_1 = leaf(0, 5)
        node_1_2_1_1_2 = leaf(1, 4)
        node_1_2_1_1 = node_1_2_1_1_1 * node_1_2_1_1_2
        node_1_2_1_2 = leaf([0, 1], 6)
        node_1_2_1 = 0.1 * node_1_2_1_1 + 0.9 * node_1_2_1_2
        node_1_2_2 = leaf(2, 3)
        node_1_2 = node_1_2_1 * node_1_2_2
        spn = 0.4 * node_1_1 + 0.6 * node_1_2

        assign_ids(spn)

        max_id = max([n.id for n in get_nodes_by_type(spn)])

        data = np.random.rand(10, 10)

        node_1_1_1_1_r = data[:, 2] * 1
        node_1_1_1_2_r = data[:, 2] * 2
        node_1_1_1_r = 0.7 * node_1_1_1_1_r + 0.3 * node_1_1_1_2_r
        node_1_1_2_r = 3 * (data[:, 0] + data[:, 1])
        node_1_1_r = node_1_1_1_r * node_1_1_2_r
        node_1_2_1_1_1_r = data[:, 0] * 5
        node_1_2_1_1_2_r = data[:, 1] * 4
        node_1_2_1_1_r = node_1_2_1_1_1_r * node_1_2_1_1_2_r
        node_1_2_1_2_r = 6 * (data[:, 0] + data[:, 1])
        node_1_2_1_r = 0.1 * node_1_2_1_1_r + 0.9 * node_1_2_1_2_r
        node_1_2_2_r = data[:, 2] * 3
        node_1_2_r = node_1_2_1_r * node_1_2_2_r
        spn_r = 0.4 * node_1_1_r + 0.6 * node_1_2_r

        self.assert_correct(spn, data, spn_r)

        lls = np.zeros((data.shape[0], max_id + 1))
        likelihood(spn, data, lls_matrix=lls)
        llls = np.zeros((data.shape[0], max_id + 1))
        log_likelihood(spn, data, lls_matrix=llls)

        self.assertTrue(np.alltrue(np.isclose(lls, np.exp(llls))))

        self.assertTrue(np.alltrue(np.isclose(spn_r, lls[:, spn.id])))
        self.assertTrue(np.alltrue(np.isclose(node_1_2_r, lls[:, node_1_2.id])))
        self.assertTrue(np.alltrue(np.isclose(node_1_2_2_r, lls[:, node_1_2_2.id])))
        self.assertTrue(np.alltrue(np.isclose(node_1_2_1_r, lls[:, node_1_2_1.id])))
        self.assertTrue(np.alltrue(np.isclose(node_1_2_1_2_r, lls[:, node_1_2_1_2.id])))
        self.assertTrue(np.alltrue(np.isclose(node_1_2_1_1_r, lls[:, node_1_2_1_1.id])))
        self.assertTrue(np.alltrue(np.isclose(node_1_2_1_1_2_r, lls[:, node_1_2_1_1_2.id])))
        self.assertTrue(np.alltrue(np.isclose(node_1_2_1_1_1_r, lls[:, node_1_2_1_1_1.id])))
        self.assertTrue(np.alltrue(np.isclose(node_1_1_r, lls[:, node_1_1.id])))
        self.assertTrue(np.alltrue(np.isclose(node_1_1_2_r, lls[:, node_1_1_2.id])))
        self.assertTrue(np.alltrue(np.isclose(node_1_1_1_r, lls[:, node_1_1_1.id])))
        self.assertTrue(np.alltrue(np.isclose(node_1_1_1_2_r, lls[:, node_1_1_1_2.id])))
        self.assertTrue(np.alltrue(np.isclose(node_1_1_1_1_r, lls[:, node_1_1_1_1.id])))


def leaf(scope, multiplier):
    l = Leaf(scope=scope)
    l.multiplier = multiplier
    return l


if __name__ == "__main__":
    unittest.main()
