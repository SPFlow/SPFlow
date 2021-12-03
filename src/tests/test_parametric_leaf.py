from spflow.base.structure.nodes.leaves.parametric import (
    Gaussian,
    LogNormal,
    MultivariateGaussian,
    Uniform,
    Bernoulli,
    Binomial,
    NegativeBinomial,
    Poisson,
    Geometric,
    Hypergeometric,
    Exponential,
    Gamma,
)
from spflow.base.inference.nodes.node import likelihood, log_likelihood
from spflow.base.structure.network_type import SPN
import numpy as np

import unittest

import random
import math


class TestParametricLeaf(unittest.TestCase):
    def test_gaussian(self):

        # ----- unit variance -----
        mean = random.random()
        var = 1.0

        gaussian = Gaussian([0], mean, math.sqrt(var))

        # create test inputs/outputs
        data = np.array([[mean], [mean + math.sqrt(var)], [mean - math.sqrt(var)]])
        targets = np.array([[0.398942], [0.241971], [0.241971]])

        probs = likelihood(gaussian, data, SPN())
        log_probs = log_likelihood(gaussian, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- larger variance -----
        mean = random.random()
        var = 5.0

        gaussian = Gaussian([0], mean, math.sqrt(var))

        # create test inputs/outputs
        data = np.array([[mean], [mean + math.sqrt(var)], [mean - math.sqrt(var)]])
        targets = np.array([[0.178412], [0.108212], [0.108212]])

        probs = likelihood(gaussian, data, SPN())
        log_probs = log_likelihood(gaussian, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- smaller variance -----
        mean = random.random()
        var = 0.2

        gaussian = Gaussian([0], mean, math.sqrt(var))

        # create test inputs/outputs
        data = np.array([[mean], [mean + math.sqrt(var)], [mean - math.sqrt(var)]])
        targets = np.array([[0.892062], [0.541062], [0.541062]])

        probs = likelihood(gaussian, data, SPN())
        log_probs = log_likelihood(gaussian, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- invalid parameters -----
        mean = random.random()

        self.assertRaises(Exception, Gaussian, [0], mean, 0.0)
        self.assertRaises(Exception, Gaussian, [0], mean, np.nextafter(0.0, -1.0))
        self.assertRaises(Exception, Gaussian, [0], np.inf, 1.0)
        self.assertRaises(Exception, Gaussian, [0], np.nan, 1.0)
        self.assertRaises(Exception, Gaussian, [0], mean, np.inf)
        self.assertRaises(Exception, Gaussian, [0], mean, np.nan)

        # set parameters to None manually
        gaussian.stdev = None
        self.assertRaises(Exception, likelihood, SPN(), gaussian, data)
        gaussian.mean = None
        self.assertRaises(Exception, likelihood, SPN(), gaussian, data)

        # invalid scope length
        self.assertRaises(Exception, Gaussian, [], 0.0, 1.0)

    def test_log_normal(self):

        # ----- configuration 1 -----
        mean = 0.0
        stdev = 0.25

        log_normal = LogNormal([0], mean, stdev)

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.0683495], [1.59577], [0.285554]])

        probs = likelihood(log_normal, data, SPN())
        log_probs = log_likelihood(log_normal, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        mean = 0.0
        stdev = 0.5

        log_normal = LogNormal([0], mean, stdev)

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.610455], [0.797885], [0.38287]])

        probs = likelihood(log_normal, data, SPN())
        log_probs = log_likelihood(log_normal, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        mean = 0.0
        stdev = 1.0

        log_normal = LogNormal([0], mean, stdev)

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.627496], [0.398942], [0.244974]])

        probs = likelihood(log_normal, data, SPN())
        log_probs = log_likelihood(log_normal, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- support -----
        mean = 0.0
        stdev = 1.0

        log_normal = LogNormal([0], mean, stdev)

        # create test inputs testing around the boundaries
        data = np.array(
            [[-1.0], [0.0], [np.inf], [np.nextafter(0.0, 1.0)], [np.finfo(np.float64).max / 3.0]]
        )

        probs = likelihood(log_normal, data, SPN())
        log_probs = log_likelihood(log_normal, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(all(np.isinf(log_probs[:3])))
        self.assertTrue(all(~np.isinf(log_probs[3:])))

        # ----- invalid parameters -----
        mean = random.random()

        self.assertRaises(Exception, LogNormal, [0], mean, 0.0)
        self.assertRaises(Exception, LogNormal, [0], mean, np.nextafter(0.0, -1.0))
        self.assertRaises(Exception, LogNormal, [0], np.inf, 1.0)
        self.assertRaises(Exception, LogNormal, [0], np.nan, 1.0)
        self.assertRaises(Exception, LogNormal, [0], mean, np.inf)
        self.assertRaises(Exception, LogNormal, [0], mean, np.nan)

        # set parameters to None manually
        log_normal.stdev = None
        self.assertRaises(Exception, likelihood, SPN(), log_normal, data)
        log_normal.mean = None
        self.assertRaises(Exception, likelihood, SPN(), log_normal, data)

        # invalid scope length
        self.assertRaises(Exception, LogNormal, [], 0.0, 1.0)

    def test_multivariate_gaussian(self):

        # ----- configuration 1 -----
        mean_vector = np.zeros(2)
        covariance_matrix = np.eye(2)

        multivariate_gaussian = MultivariateGaussian(
            [0, 1], mean_vector.tolist(), covariance_matrix.tolist()
        )

        # create test inputs/outputs
        data = np.stack([np.zeros(2), np.ones(2)], axis=0)

        targets = np.array([[0.1591549], [0.0585498]])

        probs = likelihood(multivariate_gaussian, data, SPN())
        log_probs = log_likelihood(multivariate_gaussian, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        mean_vector = np.arange(3)
        covariance_matrix = np.array(
            [
                [2, 2, 1],
                [2, 3, 2],
                [1, 2, 3],
            ]
        )

        multivariate_gaussian = MultivariateGaussian(
            [0, 1, 2], mean_vector.tolist(), covariance_matrix.tolist()
        )

        # create test inputs/outputs
        data = np.stack(
            [
                mean_vector,
                np.ones(3),
                -np.ones(3),
            ],
            axis=0,
        )

        targets = np.array([[0.0366580], [0.0159315], [0.0081795]])

        probs = likelihood(multivariate_gaussian, data, SPN())
        log_probs = log_likelihood(multivariate_gaussian, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # set parameters to None manually
        multivariate_gaussian.covariance_matrix = None
        self.assertRaises(Exception, likelihood, SPN(), multivariate_gaussian, data)
        multivariate_gaussian.mean_vector = None
        self.assertRaises(Exception, likelihood, SPN(), multivariate_gaussian, data)

        # invalid scope length
        self.assertRaises(Exception, MultivariateGaussian, [], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])

    def test_uniform(self):

        start = random.random()
        end = start + 1e-7 + random.random()

        uniform = Uniform([0], start, end)

        # create test inputs/outputs
        data = np.array(
            [
                [np.nextafter(start, -np.inf)],
                [start],
                [(start + end) / 2.0],
                [end],
                [np.nextafter(end, np.inf)],
            ]
        )
        targets = np.array(
            [[0.0], [1.0 / (end - start)], [1.0 / (end - start)], [1.0 / (end - start)], [0.0]]
        )

        probs = likelihood(uniform, data, SPN())
        log_probs = log_likelihood(uniform, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- invalid parameters -----
        start_end = random.random()

        self.assertRaises(Exception, Uniform, [0], start_end, start_end)
        self.assertRaises(Exception, Uniform, [0], start_end, np.nextafter(start_end, -1.0))
        self.assertRaises(Exception, Uniform, [0], np.inf, 0.0)
        self.assertRaises(Exception, Uniform, [0], np.nan, 0.0)
        self.assertRaises(Exception, Uniform, [0], 0.0, np.inf)
        self.assertRaises(Exception, Uniform, [0], 0.0, np.nan)

        # set parameters to None manually
        uniform.end = None
        self.assertRaises(Exception, likelihood, SPN(), uniform, data)
        uniform.start = None
        self.assertRaises(Exception, likelihood, SPN(), uniform, data)

        # invalid scope length
        self.assertRaises(Exception, Uniform, [], 0.0, 1.0)

    def test_bernoulli(self):

        p = random.random()

        bernoulli = Bernoulli([0], p)

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[1.0 - p], [p]])

        probs = likelihood(bernoulli, data, SPN())
        log_probs = log_likelihood(bernoulli, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- (invalid) parameters -----

        # p = 0
        bernoulli = Bernoulli([0], 0.0)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[1.0], [0.0]])

        probs = likelihood(
            bernoulli,
            data,
            SPN(),
        )
        log_probs = log_likelihood(bernoulli, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # p = 1
        bernoulli = Bernoulli([0], 1.0)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[0.0], [1.0]])

        probs = likelihood(bernoulli, data, SPN())
        log_probs = log_likelihood(bernoulli, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # p < 0 and p > 1
        self.assertRaises(Exception, Bernoulli, [0], np.nextafter(1.0, 2.0))
        self.assertRaises(Exception, Bernoulli, [0], np.nextafter(0.0, -1.0))

        # inf, nan
        self.assertRaises(Exception, Bernoulli, [0], np.inf)
        self.assertRaises(Exception, Bernoulli, [0], np.nan)

        # set parameters to None manually
        bernoulli.p = None
        self.assertRaises(Exception, likelihood, SPN(), bernoulli, data)

        # invalid scope length
        self.assertRaises(Exception, Bernoulli, [], 0.5)

        # ----- support -----
        p = random.random()

        bernoulli = Bernoulli([0], p)

        data = np.array([[np.nextafter(0.0, -1.0)], [0.5], [np.nextafter(1.0, 2.0)]])

        probs = likelihood(bernoulli, data, SPN())
        log_probs = log_likelihood(bernoulli, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(all(probs == 0.0))

    def test_binomial(self):

        # ----- configuration 1 -----
        n = 10
        p = 0.5

        binomial = Binomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [5], [10]])
        targets = np.array([[0.000976563], [0.246094], [0.000976563]])

        probs = likelihood(binomial, data, SPN())
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        n = 5
        p = 0.8

        binomial = Binomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.00032], [0.0512], [0.32768]])

        probs = likelihood(
            binomial,
            data,
            SPN(),
        )
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        n = 15
        p = 0.3

        binomial = Binomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [7], [15]])
        targets = np.array([[0.00474756], [0.08113], [0.0000000143489]])

        probs = likelihood(
            binomial,
            data,
            SPN(),
        )
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- (invalid) parameters -----

        # p = 0
        binomial = Binomial([0], 1, 0.0)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[1.0], [0.0]])

        probs = likelihood(binomial, data, SPN())
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # p = 1
        binomial = Binomial([0], 1, 1.0)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[0.0], [1.0]])

        probs = likelihood(
            binomial,
            data,
            SPN(),
        )
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # p < 0 and p > 1
        self.assertRaises(Exception, Binomial, [0], 1, np.nextafter(1.0, 2.0))
        self.assertRaises(Exception, Binomial, [0], 1, np.nextafter(0.0, -1.0))

        # n = 0
        binomial = Binomial([0], 0, 0.5)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[1.0], [0.0]])

        probs = likelihood(binomial, data, SPN())
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # n < 0
        self.assertRaises(Exception, Binomial, [0], -1, 0.5)

        # TODO: n float

        # inf, nan
        self.assertRaises(Exception, Binomial, [0], np.inf, 0.5)
        self.assertRaises(Exception, Binomial, [0], np.nan, 0.5)
        self.assertRaises(Exception, Binomial, [0], 1, np.inf)
        self.assertRaises(Exception, Binomial, [0], 1, np.nan)

        # set parameters to None manually
        binomial.p = None
        self.assertRaises(Exception, likelihood, SPN(), binomial, data)
        binomial.n = None
        self.assertRaises(Exception, likelihood, SPN(), binomial, data)

        # invalid scope length
        self.assertRaises(Exception, Binomial, [], 1, 0.5)

        # ----- support -----

        binomial = Binomial([0], 1, 0.0)

        data = np.array([[-1.0], [2.0]])

        probs = likelihood(binomial, data, SPN())
        log_probs = log_likelihood(binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(all(probs == 0))

    def test_negative_binomial(self):

        # ----- configuration 1 -----
        n = 10
        p = 0.4

        negative_binomial = NegativeBinomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [5], [10]])
        targets = np.array([[0.000104858], [0.0163238], [0.0585708]])

        probs = likelihood(negative_binomial, data, SPN())
        log_probs = log_likelihood(negative_binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        n = 20
        p = 0.3

        negative_binomial = NegativeBinomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [10], [20]])
        targets = np.array([[0.0000000000348678], [0.0000197282], [0.00191757]])

        probs = likelihood(negative_binomial, data, SPN())
        log_probs = log_likelihood(negative_binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- (invalid) parameters -----

        # p = 1
        negative_binomial = NegativeBinomial([0], 1, 1.0)

        data = np.array([[0.0], [1.0]])
        targets = np.array([[1.0], [0.0]])

        probs = likelihood(negative_binomial, data, SPN())
        log_probs = log_likelihood(negative_binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # p = 0
        self.assertRaises(Exception, NegativeBinomial, [0], 1, 0.0)

        # 1 >= p > 0
        NegativeBinomial([0], 1, np.nextafter(0.0, 1.0))

        # p < 0 and p > 1
        self.assertRaises(Exception, Binomial, [0], 1, np.nextafter(1.0, 2.0))
        self.assertRaises(Exception, Binomial, [0], 1, np.nextafter(0.0, -1.0))

        # p inf, nan
        self.assertRaises(Exception, NegativeBinomial, [0], 1, np.inf)
        self.assertRaises(Exception, NegativeBinomial, [0], 1, np.nan)

        # n = 0
        NegativeBinomial([0], 0.0, 1.0)

        # n < 0
        self.assertRaises(Exception, NegativeBinomial, [0], np.nextafter(0.0, -1.0), 1.0)

        # n inf, nan
        self.assertRaises(Exception, NegativeBinomial, [0], np.inf, 1.0)
        self.assertRaises(Exception, NegativeBinomial, [0], np.nan, 1.0)

        # TODO: n float

        # set parameters to None manually
        negative_binomial.p = None
        self.assertRaises(Exception, likelihood, SPN(), negative_binomial, data)
        negative_binomial.n = None
        self.assertRaises(Exception, likelihood, SPN(), negative_binomial, data)

        # invalid scope length
        self.assertRaises(Exception, NegativeBinomial, [], 1, 0.5)

        # ----- support -----
        n = 20
        p = 0.3

        negative_binomial = NegativeBinomial([0], n, p)

        data = np.array([[np.nextafter(0.0, -1.0)], [0.0]])

        probs = likelihood(negative_binomial, data, SPN())
        log_probs = log_likelihood(negative_binomial, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(all(probs[0] == 0.0))
        self.assertTrue(all(probs[1] != 0.0))

    def test_poisson(self):

        # ----- configuration 1 -----
        l = 1

        poisson = Poisson([0], l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.367879], [0.18394], [0.00306566]])

        probs = likelihood(poisson, data, SPN())
        log_probs = log_likelihood(poisson, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        l = 4

        poisson = Poisson([0], l)

        # create test inputs/outputs
        data = np.array([[2], [4], [10]])
        targets = np.array([[0.146525], [0.195367], [0.00529248]])

        probs = likelihood(poisson, data, SPN())
        log_probs = log_likelihood(poisson, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        l = 10

        poisson = Poisson([0], l)

        # create test inputs/outputs
        data = np.array([[5], [10], [15]])
        targets = np.array([[0.0378333], [0.12511], [0.0347181]])

        probs = likelihood(poisson, data, SPN())
        log_probs = log_likelihood(poisson, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- invalid parameters -----
        self.assertRaises(Exception, Poisson, [0], -np.inf)
        self.assertRaises(Exception, Poisson, [0], np.inf)
        self.assertRaises(Exception, Poisson, [0], np.nan)

        # set parameters to None manually
        poisson.l = None
        self.assertRaises(Exception, likelihood, SPN(), poisson, data)

        # invalid scope length
        self.assertRaises(Exception, Poisson, [], 1)

        # ----- support -----

        l = random.random()

        poisson = Poisson([0], l)

        # create test inputs/outputs
        data = np.array([[-1.0], [-0.5], [0.0]])

        probs = likelihood(poisson, data, SPN())
        log_probs = log_likelihood(poisson, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.all(probs[:2] == 0))
        self.assertTrue(np.all(probs[-1] != 0))

    def test_geometric(self):

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

        # ----- invalid parameters -----

        # p = 0
        self.assertRaises(Exception, Geometric, [0], 0.0)
        self.assertRaises(Exception, Geometric, [0], np.inf)
        self.assertRaises(Exception, Geometric, [0], np.nan)

        # set parameters to None manually
        geometric.p = None
        self.assertRaises(Exception, likelihood, SPN(), geometric, data)

        # invalid scope length
        self.assertRaises(Exception, Geometric, [], 0.5)

        # ----- support -----

        p = 0.8

        geometric = Geometric([0], p)

        # create test inputs/outputs
        data = np.array([[0], [np.nextafter(1.0, 0.0)], [1.5], [1]])

        probs = likelihood(geometric, data, SPN())
        log_probs = log_likelihood(geometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.all(probs[:3] == 0))
        self.assertTrue(np.all(probs[-1] != 0))

    def test_hypergeometric(self):

        # ----- configuration 1 -----
        N = 500
        M = 100
        n = 50

        hypergeometric = Hypergeometric([0], N, M, n)

        # create test inputs/outputs
        data = np.array([[5], [10], [15]])
        targets = np.array([[0.0257071], [0.147368], [0.0270206]])

        probs = likelihood(hypergeometric, data, SPN())
        log_probs = log_likelihood(hypergeometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        N = 100
        M = 50
        n = 10

        hypergeometric = Hypergeometric([0], N, M, n)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.00723683], [0.259334], [0.00059342]])

        probs = likelihood(hypergeometric, data, SPN())
        log_probs = log_likelihood(hypergeometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- support -----
        N = 15
        M = 10
        n = 10

        hypergeometric = Hypergeometric([0], N, M, n)

        # create test inputs/outputs
        data = np.array([[4], [11], [5], [10]])

        probs = likelihood(hypergeometric, data, SPN())
        log_probs = log_likelihood(hypergeometric, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(all(probs[:2] == 0))
        self.assertTrue(all(probs[2:] != 0))

        # ----- invalid parameters -----
        self.assertRaises(Exception, Hypergeometric, -1, 1, 1)
        self.assertRaises(Exception, Hypergeometric, 1, -1, 1)
        self.assertRaises(Exception, Hypergeometric, 1, 2, 1)
        self.assertRaises(Exception, Hypergeometric, 1, 1, -1)
        self.assertRaises(Exception, Hypergeometric, 1, 1, 2)
        self.assertRaises(Exception, Hypergeometric, [0], np.inf, 1, 1)
        self.assertRaises(Exception, Hypergeometric, [0], np.nan, 1, 1)
        self.assertRaises(Exception, Hypergeometric, [0], 1, np.inf, 1)
        self.assertRaises(Exception, Hypergeometric, [0], 1, np.nan, 1)
        self.assertRaises(Exception, Hypergeometric, [0], 1, 1, np.inf)
        self.assertRaises(Exception, Hypergeometric, [0], 1, 1, np.nan)

        # set parameters to None manually
        hypergeometric.n = None
        self.assertRaises(Exception, likelihood, SPN(), hypergeometric, data)
        hypergeometric.M = None
        self.assertRaises(Exception, likelihood, SPN(), hypergeometric, data)
        hypergeometric.N = None
        self.assertRaises(Exception, likelihood, SPN(), hypergeometric, data)

        # invalid scope length
        self.assertRaises(Exception, Hypergeometric, [], 1, 1, 1)

    def test_exponential(self):

        # ----- configuration 1 -----
        l = 0.5

        exponential = Exponential([0], l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.5], [0.18394], [0.0410425]])

        probs = likelihood(exponential, data, SPN())
        log_probs = log_likelihood(exponential, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        l = 1.0

        exponential = Exponential([0], l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[1.0], [0.135335], [0.00673795]])

        probs = likelihood(exponential, data, SPN())
        log_probs = log_likelihood(exponential, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        l = 1.5

        exponential = Exponential([0], l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[1.5], [0.0746806], [0.000829627]])

        probs = likelihood(exponential, data, SPN())
        log_probs = log_likelihood(exponential, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- (invalid) parameters -----
        Exponential([0], np.nextafter(0.0, 1.0))
        self.assertRaises(Exception, Exponential, [0], 0.0)
        self.assertRaises(Exception, Exponential, [0], -1.0)
        self.assertRaises(Exception, Exponential, [0], np.inf)
        self.assertRaises(Exception, Exponential, [0], np.nan)

        # set parameters to None manually
        exponential.l = None
        self.assertRaises(Exception, likelihood, SPN(), exponential, data)

        # invalid scope length
        self.assertRaises(Exception, Exponential, [], 1.0)

        # ----- support -----
        l = 1.5

        exponential = Exponential([0], l)

        # create test inputs/outputs
        data = np.array([[np.nextafter(0.0, -1.0)], [0.0]])

        probs = likelihood(exponential, data, SPN())
        log_probs = log_likelihood(exponential, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(all(probs[0] == 0.0))
        self.assertTrue(all(probs[1] != 0.0))

    def test_gamma(self):

        # ----- configuration 1 -----
        alpha = 1.0
        beta = 1.0

        gamma = Gamma([0], alpha, beta)

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.904837], [0.367879], [0.0497871]])

        probs = likelihood(gamma, data, SPN())
        log_probs = log_likelihood(gamma, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        alpha = 2.0
        beta = 2.0

        gamma = Gamma([0], alpha, beta)

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.327492], [0.541341], [0.029745]])

        probs = likelihood(gamma, data, SPN())
        log_probs = log_likelihood(gamma, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        alpha = 2.0
        beta = 1.0

        gamma = Gamma([0], alpha, beta)

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.0904837], [0.367879], [0.149361]])

        probs = likelihood(gamma, data, SPN())
        log_probs = log_likelihood(gamma, data, SPN())

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- (invalid) parameters -----
        Gamma([0], np.nextafter(0.0, 1.0), 1.0)
        Gamma([0], 1.0, np.nextafter(0.0, 1.0))
        self.assertRaises(Exception, Gamma, [0], np.nextafter(0.0, -1.0), 1.0)
        self.assertRaises(Exception, Gamma, [0], 1.0, np.nextafter(0.0, -1.0))
        self.assertRaises(Exception, Gamma, [0], np.inf, 1.0)
        self.assertRaises(Exception, Gamma, [0], np.nan, 1.0)
        self.assertRaises(Exception, Gamma, [0], 1.0, np.inf)
        self.assertRaises(Exception, Gamma, [0], 1.0, np.nan)

        # set parameters to None manually
        gamma.beta = None
        self.assertRaises(Exception, likelihood, SPN(), gamma, data)
        gamma.alpha = None
        self.assertRaises(Exception, likelihood, SPN(), gamma, data)

        # invalid scope length
        self.assertRaises(Exception, Gamma, [], 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
