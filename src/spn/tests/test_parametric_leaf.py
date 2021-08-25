from spn.python.structure.nodes.leaves.parametric.parametric import (
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
from spn.python.inference.nodes.node import likelihood, log_likelihood
import numpy as np

import random

import unittest

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

        probs = likelihood(gaussian, data)
        log_probs = log_likelihood(gaussian, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- larger variance -----
        mean = random.random()
        var = 5.0

        gaussian = Gaussian([0], mean, math.sqrt(var))

        # create test inputs/outputs
        data = np.array([[mean], [mean + math.sqrt(var)], [mean - math.sqrt(var)]])
        targets = np.array([[0.178412], [0.108212], [0.108212]])

        probs = likelihood(gaussian, data)
        log_probs = log_likelihood(gaussian, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- smaller variance -----
        mean = random.random()
        var = 0.2

        gaussian = Gaussian([0], mean, math.sqrt(var))

        # create test inputs/outputs
        data = np.array([[mean], [mean + math.sqrt(var)], [mean - math.sqrt(var)]])
        targets = np.array([[0.892062], [0.541062], [0.541062]])

        probs = likelihood(gaussian, data)
        log_probs = log_likelihood(gaussian, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_log_normal(self):

        # ----- configuration 1 -----
        mean = 0.0
        stdev = 0.25

        log_normal = LogNormal([0], mean, stdev)

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.0683495], [1.59577], [0.285554]])

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        mean = 0.0
        stdev = 0.5

        log_normal = LogNormal([0], mean, stdev)

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.610455], [0.797885], [0.38287]])

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        mean = 0.0
        stdev = 1.0

        log_normal = LogNormal([0], mean, stdev)

        # create test inputs/outputs
        data = np.array([[0.5], [1.0], [1.5]])
        targets = np.array([[0.627496], [0.398942], [0.244974]])

        probs = likelihood(log_normal, data)
        log_probs = log_likelihood(log_normal, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_multivariate_gaussian(self):
        pass

    def test_uniform(self):
        
        start = random.random()
        end = start + 1e-7 + random.random()

        uniform = Uniform([0], start, end)

        # create test inputs/outputs
        data = np.array([[np.nextafter(start, -np.inf)], [start], [(start+end)/2.0], [end], [np.nextafter(end, np.inf)]])
        targets = np.array([[0.0], [1.0/(end-start)], [1.0/(end-start)], [1.0/(end-start)], [0.0]])

        probs = likelihood(uniform, data)
        log_probs = log_likelihood(uniform, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_bernoulli(self):

        p = random.random()

        bernoulli = Bernoulli([0], p)

        # create test inputs/outputs
        data = np.array([[0], [1]])
        targets = np.array([[1.0 - p], [p]])

        probs = likelihood(bernoulli, data)
        log_probs = log_likelihood(bernoulli, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_binomial(self):

        # ----- configuration 1 -----
        n = 10
        p = 0.5

        binomial = Binomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [5], [10]])
        targets = np.array([[0.000976563], [0.246094], [0.000976563]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        n = 5
        p = 0.8

        binomial = Binomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.00032], [0.0512], [0.32768]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        n = 15
        p = 0.3

        binomial = Binomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [7], [15]])
        targets = np.array([[0.00474756], [0.08113], [0.0000000143489]])

        probs = likelihood(binomial, data)
        log_probs = log_likelihood(binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_negative_binomial(self):

        # ----- configuration 1 -----
        n = 10
        p = 0.4

        negative_binomial = NegativeBinomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [5], [10]])
        targets = np.array([[0.000104858], [0.0163238], [0.0585708]])

        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        n = 20
        p = 0.3

        negative_binomial = NegativeBinomial([0], n, p)

        # create test inputs/outputs
        data = np.array([[0], [10], [20]])
        targets = np.array([[0.0000000000348678], [0.0000197282], [0.00191757]])

        probs = likelihood(negative_binomial, data)
        log_probs = log_likelihood(negative_binomial, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_poisson(self):

        # ----- configuration 1 -----
        l = 1

        poisson = Poisson([0], l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.367879], [0.18394], [0.00306566]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        l = 4

        poisson = Poisson([0], l)

        # create test inputs/outputs
        data = np.array([[2], [4], [10]])
        targets = np.array([[0.146525], [0.195367], [0.00529248]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        l = 10

        poisson = Poisson([0], l)

        # create test inputs/outputs
        data = np.array([[5], [10], [15]])
        targets = np.array([[0.0378333], [0.12511], [0.0347181]])

        probs = likelihood(poisson, data)
        log_probs = log_likelihood(poisson, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_geometric(self):

        # ----- configuration 1 -----
        p = 0.2

        geometric = Geometric([0], p)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.2], [0.08192], [0.0268435]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        p = 0.5

        geometric = Geometric([0], p)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.5], [0.03125], [0.000976563]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        p = 0.8

        geometric = Geometric([0], p)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.8], [0.00128], [0.0000004096]])

        probs = likelihood(geometric, data)
        log_probs = log_likelihood(geometric, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_hypergeometric(self):

        # ----- configuration 1 -----
        M = 500
        N = 50
        n = 100

        hypergeometric = Hypergeometric([0], M=M, N=N, n=n)

        # create test inputs/outputs
        data = np.array([[5], [10], [15]])
        targets = np.array([[0.0257071], [0.147368], [0.0270206]])

        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        M = 100
        N = 10
        n = 50

        hypergeometric = Hypergeometric([0], M=M, N=N, n=n)

        # create test inputs/outputs
        data = np.array([[1], [5], [10]])
        targets = np.array([[0.00723683], [0.259334], [0.00059342]])

        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        M = 250
        N = 100
        n = 150

        hypergeometric = Hypergeometric([0], M=M, N=N, n=n)

        # create test inputs/outputs
        data = np.array([[40], [50], [60]])
        targets = np.array([[0.0000000999597], [0.0033495], [0.10478]])

        probs = likelihood(hypergeometric, data)
        log_probs = log_likelihood(hypergeometric, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_exponential(self):

        # ----- configuration 1 -----
        l = 0.5

        exponential = Exponential([0], l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[0.5], [0.18394], [0.0410425]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        l = 1.0

        exponential = Exponential([0], l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[1.0], [0.135335], [0.00673795]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        l = 1.5

        exponential = Exponential([0], l)

        # create test inputs/outputs
        data = np.array([[0], [2], [5]])
        targets = np.array([[1.5], [0.0746806], [0.000829627]])

        probs = likelihood(exponential, data)
        log_probs = log_likelihood(exponential, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

    def test_gamma(self):

        # ----- configuration 1 -----
        alpha = 1.0
        beta = 1.0

        gamma = Gamma([0], alpha, beta)

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.904837], [0.367879], [0.0497871]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 2 -----
        alpha = 2.0
        beta = 2.0

        gamma = Gamma([0], alpha, beta)

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.327492], [0.541341], [0.029745]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))

        # ----- configuration 3 -----
        alpha = 2.0
        beta = 1.0

        gamma = Gamma([0], alpha, beta)

        # create test inputs/outputs
        data = np.array([[0.1], [1.0], [3.0]])
        targets = np.array([[0.0904837], [0.367879], [0.149361]])

        probs = likelihood(gamma, data)
        log_probs = log_likelihood(gamma, data)

        self.assertTrue(np.allclose(probs, np.exp(log_probs)))
        self.assertTrue(np.allclose(probs, targets))


if __name__ == "__main__":
    unittest.main()
