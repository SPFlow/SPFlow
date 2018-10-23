import unittest

from numpy.random.mtrand import RandomState

from spn.algorithms.Posteriors import *
from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.parametric.Sampling import sample_parametric_node


class TestPosterior(unittest.TestCase):
    def test_Posterior(self):
        def update_params(generator, ll_node, prior, n=1000000):
            ll_node.row_ids = list(range(n))
            ll_node.scope = [0]
            X = sample_parametric_node(generator, n, None, RandomState(1234)).reshape(-1, 1)
            update_parametric_parameters_posterior(ll_node, X, RandomState(1234), prior)
            print("expected", generator.params, "found", ll_node.params)
            return generator, ll_node

        generator, node = update_params(generator=Gaussian(mean=40, stdev=5), ll_node=Gaussian(mean=0, stdev=1),
                                        prior=PriorNormalInverseGamma(m_0=1, V_0=1, a_0=1, b_0=1))
        self.assertAlmostEqual(generator.mean, node.mean, 1)
        self.assertAlmostEqual(generator.stdev, node.stdev, 1)

        generator, node = update_params(generator=Gamma(alpha=3, beta=2), ll_node=Gamma(alpha=3, beta=0.1),
                                        prior=PriorGamma(a_0=1, b_0=1))
        self.assertAlmostEqual(generator.beta, node.beta, 1)

        generator, node = update_params(generator=LogNormal(mean=10, stdev=1.25), ll_node=LogNormal(mean=0, stdev=1),
                                        prior=PriorNormal(mu_0=1, tau_0=1))
        self.assertAlmostEqual(generator.mean, node.mean, 1)

        generator, node = update_params(generator=Poisson(mean=15), ll_node=Poisson(mean=1),
                                        prior=PriorGamma(a_0=1, b_0=1))
        self.assertAlmostEqual(generator.mean, node.mean, 1)

        generator, node = update_params(generator=Bernoulli(p=0.1), ll_node=Bernoulli(p=0.8),
                                        prior=PriorBeta(a_0=1, b_0=1))
        self.assertAlmostEqual(generator.p, node.p, 1)

        generator, node = update_params(generator=Geometric(p=0.1), ll_node=Geometric(p=0.8),
                                        prior=PriorBeta(a_0=1, b_0=1))
        self.assertAlmostEqual(generator.p, node.p, 1)

        generator, node = update_params(generator=Exponential(l=0.1), ll_node=Exponential(l=0.8),
                                        prior=PriorGamma(a_0=1, b_0=1))
        self.assertAlmostEqual(generator.l, node.l, 1)

        generator, node = update_params(generator=Categorical(p=[0.1, 0.5, 0.4]),
                                        ll_node=Categorical(p=[0.4, 0.3, 0.3]),
                                        prior=PriorDirichlet(alphas_0=0.1))
        self.assertTrue(np.alltrue(np.isclose(generator.p, node.p, 0.01)))


if __name__ == '__main__':
    unittest.main()
