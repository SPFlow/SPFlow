import unittest

from numpy.random.mtrand import RandomState

from spn.algorithms.Inference import add_node_likelihood, log_likelihood
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import assign_ids, Leaf, Sum, get_nodes_by_type

from scipy.stats import chisquare

import numpy as np

from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
from spn.structure.leaves.parametric.Sampling import add_parametric_sampling_support


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
        spn = 0.5 * (Gaussian(mean=10, stdev=0.000000001, scope=0) * Categorical(p=[1.0, 0], scope=1)) + 0.5 * (
            Gaussian(mean=50, stdev=0.000000001, scope=0) * Categorical(p=[0, 1.0], scope=1)
        )

        rand_gen = np.random.RandomState(17)

        data = np.zeros((2, 2))

        data[1, 1] = 1

        data[:, 0] = np.nan

        samples = sample_instances(spn, data, rand_gen)

        self.assertAlmostEqual(samples[0, 0], 10)
        self.assertAlmostEqual(samples[1, 0], 50)

    def test_dag_spns(self):
        D = Gaussian(mean=0.0, stdev=1.0, scope=[0])
        E = Gaussian(mean=10.0, stdev=1.0, scope=[0])
        F = Gaussian(mean=20.0, stdev=1.0, scope=[0])

        B = 0.5 * D + 0.5 * E
        C = 0.5 * E + 0.5 * F

        H = 0.5 * D + 0.5 * E
        I = 0.5 * D + 0.5 * E

        G = 0.5 * H + 0.5 * I
        A = 0.5 * B + 0.5 * C
        Z = 0.5 * A + 0.5 * G

        data = np.zeros((2000, 1))

        data[:, 0] = np.nan

        samples = sample_instances(Z, data, np.random.RandomState(17))


if __name__ == "__main__":
    unittest.main()
