import unittest

from numpy.random.mtrand import RandomState

from spn.algorithms.Inference import add_node_likelihood, log_likelihood
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import assign_ids, Leaf, Sum, get_nodes_by_type

from scipy.stats import chisquare

import numpy as np


def constant_equal_ll(node, data, dtype=np.float64, node_likelihood=None):
    probs = np.zeros((data.shape[0], 1), dtype=dtype)
    probs[:] = 0.5
    return probs


def node_fixed_ll(node, data, dtype=np.float64, node_likelihood=None):
    probs = np.zeros((data.shape[0], 1), dtype=dtype)
    probs[:] = node.prob
    return np.log(probs)


def leaf(prob, scope=0):
    r = Leaf(scope)
    r.prob = prob
    return r


class TestSampling(unittest.TestCase):
    def test_correct_parameters(self):
        node_1_2_2 = Leaf(0)
        node_1_2_1 = Leaf(1)
        node_1_1 = Leaf([0, 1])
        node_1_2 = node_1_2_1 * node_1_2_2
        spn = 0.1 * node_1_1 + 0.9 * node_1_2
        node_1_2.id = 0

        rand_gen = RandomState(1234)
        with self.assertRaises(AssertionError):
            sample_instances(spn, rand_gen.rand(10, 3), rand_gen)

        assign_ids(spn)
        node_1_2_2.id += 1

        with self.assertRaises(AssertionError):
            sample_instances(spn, rand_gen.rand(10, 3), rand_gen)

    def test_induced_trees(self):
        add_node_likelihood(Leaf, constant_equal_ll)

        n = 100000

        spn = 0.1 * (Leaf(0) * Leaf(1)) + 0.9 * (Leaf(0) * Leaf(1))

        rand_gen = np.random.RandomState(17)

        data = rand_gen.rand(10, 2)

        data[:, 0] = np.nan

        sample_instances(spn, data, rand_gen)


if __name__ == '__main__':
    unittest.main()
