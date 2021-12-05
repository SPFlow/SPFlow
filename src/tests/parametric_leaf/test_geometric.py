from spflow.base.structure.nodes.leaves.parametric import Geometric
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest


class TestGeometric(unittest.TestCase):
    def test_likelihood(self):

        # ----- configuration 1 -----
        p = 0.2

        geometric = Geometric([0], p)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.2], [0.08192], [0.0268435]])

        probs = likelihood(geometric, data, SPN())
        log_probs = log_likelihood(geometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        p = 0.5

        geometric = Geometric([0], p)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.5], [0.03125], [0.000976563]])

        probs = likelihood(geometric, data, SPN())
        log_probs = log_likelihood(geometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        p = 0.8

        geometric = Geometric([0], p)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.8], [0.00128], [0.0000004096]])

        probs = likelihood(geometric, data, SPN())
        log_probs = log_likelihood(geometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_initialization(self):

        # p = 0
        self.assertRaises(Exception, Geometric, [0], 0.0)
        self.assertRaises(Exception, Geometric, [0], np.inf)
        self.assertRaises(Exception, Geometric, [0], np.nan)

        geometric = Geometric([0], 0.5)
        data = np.array([[1], [5], [10]])

        # set parameters to None manually
        geometric.p = None
        self.assertRaises(Exception, likelihood, SPN(), geometric, data)

        # invalid scope lengths
        self.assertRaises(Exception, Geometric, [], 0.5)
        self.assertRaises(Exception, Geometric, [0,1], 0.5)

    def test_support(self):

        p = 0.8

        geometric = Geometric([0], p)

        # create test inputs/outputs
        data = np.array([[0], [np.nextafter(1.0, 0.0)], [1.5], [1]])

        probs = likelihood(geometric, data, SPN())
        log_probs = log_likelihood(geometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.all(probs[:3] == 0))
        self.assertTrue(np.all(probs[-1] != 0))


if __name__ == "__main__":
    unittest.main()
