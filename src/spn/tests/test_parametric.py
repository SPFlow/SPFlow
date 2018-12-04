import unittest

from spn.algorithms.Inference import likelihood, log_likelihood
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import *


class TestParametric(unittest.TestCase):
    def setUp(self):
        add_parametric_inference_support()
        self.tested = set()

    def assert_correct(self, node, x, result):
        self.tested.add(type(node))

        data = np.array([x], dtype=np.float).reshape(-1, 1)
        node.scope = [0]
        l = likelihood(node, data)
        self.assertAlmostEqual(result, l[0, 0], 5)
        self.assertTrue(np.alltrue(np.isclose(np.log(l), log_likelihood(node, data))))

        data = np.random.rand(10, 10)
        data[:, 5] = x
        node.scope = [5]
        l = likelihood(node, data)
        self.assertEqual(l.shape[0], data.shape[0])
        self.assertEqual(l.shape[1], 1)
        self.assertTrue(np.isclose(np.var(l), 0))
        self.assertTrue(np.alltrue(np.isclose(result, l[0, 0])))
        self.assertTrue(np.alltrue(np.isclose(np.log(l), log_likelihood(node, data))))

    def test_Parametric_inference(self):
        # N[PDF[NormalDistribution[4, 1], 5], 6] = 0.241971
        self.assert_correct(Gaussian(mean=4, stdev=1), 5, 0.241971)
        # N[PDF[NormalDistribution[10, 0.5], 9], 6] = 0.107982
        self.assert_correct(Gaussian(mean=10, stdev=0.5), 9, 0.107982)

        # N[PDF[GammaDistribution[4, 1], 4], 6] = 0.195367
        self.assert_correct(Gamma(4.0, 1.0), 4, 0.195367)
        # N[PDF[GammaDistribution[10, 0.5], 4], 6] = 0.248154
        self.assert_correct(Gamma(10.0, 1 / 0.5), 4, 0.248154)

        # N[PDF[PoissonDistribution[2], 3], 6] = 0.180447
        self.assert_correct(Poisson(2.0), 3, 0.180447)
        # N[PDF[PoissonDistribution[6.5], 4], 6] = 0.111822
        self.assert_correct(Poisson(6.5), 4, 0.111822)

        # N[PDF[ExponentialDistribution[1.5], 1], 6] = 0.334695
        self.assert_correct(Exponential(1.5), 1, 0.334695)

        # not comparable to mathematica
        self.assert_correct(Geometric(0.8), 1, 0.8)
        self.assert_correct(Geometric(0.8), 2, 0.8 * 0.2)

        # N[PDF[EmpiricalDistribution[{1/3, 1/2, 1/6} -> {0, 1, 2}], 0], 6] = 0.333333
        self.assert_correct(Categorical([1 / 3, 1 / 2, 1 / 6]), 0, 0.333333)
        self.assert_correct(Categorical([1 / 3, 1 / 2, 1 / 6]), 1, 0.5)
        self.assert_correct(Categorical([1 / 3, 1 / 2, 1 / 6]), 2, 0.166667)

        # N[PDF[BernoulliDistribution[0.25], 0], 6] = 0.75
        self.assert_correct(Bernoulli(0.25), 0, 0.75)

        # N[PDF[LogNormalDistribution[0, 0.25], 1], 6]
        self.assert_correct(LogNormal(mean=0, stdev=0.25), 1.0, 1.59577)

        # N[PDF[NegativeBinomialDistribution[2, 0.2], 5], 6]
        self.assert_correct(LogNormal(mean=0, stdev=0.25), 1.0, 1.59577)

        for child in Parametric.__subclasses__():
            if child not in self.tested:
                print("not tested", child)

    def test_Parametric_constructors_from_data(self):
        pass


if __name__ == "__main__":
    unittest.main()
