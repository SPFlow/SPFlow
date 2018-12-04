import unittest

from scipy.stats import *

from spn.algorithms.Inference import likelihood
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.parametric.Sampling import sample_parametric_node
from spn.structure.leaves.parametric.utils import get_scipy_obj_params
import os


class TestParametricSampling(unittest.TestCase):
    def setUp(self):
        add_parametric_inference_support()

    def assert_correct_node_sampling_continuous(self, node, samples, plot):
        node.scope = [0]
        rand_gen = np.random.RandomState(1234)
        samples_gen = sample_parametric_node(node, 1000000, None, rand_gen)

        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1)

            x = np.linspace(np.min(samples), np.max(samples), 1000)
            ax.plot(
                x, likelihood(node, x.reshape(-1, 1)), "r-", lw=2, alpha=0.6, label=node.__class__.__name__ + " pdf"
            )
            ax.hist(samples, normed=True, histtype="stepfilled", alpha=0.7, bins=1000)
            ax.legend(loc="best", frameon=False)
            plt.show()

        scipy_obj, params = get_scipy_obj_params(node)
        # H_0 dist are identical
        test_outside_samples = kstest(samples, lambda x: scipy_obj.cdf(x, **params))
        # reject H_0 (dist are identical) if p < 0.05
        # we pass the test if they are identical, pass if p >= 0.05
        self.assertGreaterEqual(test_outside_samples.pvalue, 0.05)

        test_generated_samples = kstest(samples_gen, lambda x: scipy_obj.cdf(x, **params))
        # reject H_0 (dist are identical) if p < 0.05
        # we pass the test if they are identical, pass if p >= 0.05
        self.assertGreaterEqual(test_generated_samples.pvalue, 0.05)

    def assert_correct_node_sampling_discrete(self, node, samples, plot):
        node.scope = [0]
        rand_gen = np.random.RandomState(1234)
        samples_gen = sample_parametric_node(node, 1000000, None, rand_gen)

        fvals, fobs = np.unique(samples, return_counts=True)

        # H_0 data comes from same dist
        test_outside_samples = chisquare(fobs, (likelihood(node, fvals.reshape(-1, 1)) * samples.shape[0])[:, 0])
        # reject H_0 (data comes from dist) if p < 0.05
        # we pass the test if they come from the dist, pass if p >= 0.05
        self.assertGreaterEqual(test_outside_samples.pvalue, 0.05)

        fvals, fobs = np.unique(samples_gen, return_counts=True)

        test_generated_samples = chisquare(fobs, (likelihood(node, fvals.reshape(-1, 1)) * samples.shape[0])[:, 0])
        # reject H_0 (data comes from dist) if p < 0.05
        # we pass the test if they come from the dist, pass if p >= 0.05
        self.assertGreaterEqual(test_generated_samples.pvalue, 0.05)

    def test_Parametrics(self):
        # this test loads datasets generated in R
        # then does a quick and dirty goodness of fit test to see if that data came from the given distribution
        # and then generates data and applies the same goodness of fit test, checking that it came from the same given distribution
        fpath = os.path.dirname(os.path.abspath(__file__)) + "/"
        geom_samples = np.loadtxt(fpath + "parametric_samples/geom_prob0.7.csv", skiprows=1)
        self.assert_correct_node_sampling_discrete(Geometric(p=0.7), geom_samples + 1, False)

        pois_samples = np.loadtxt(fpath + "parametric_samples/pois_lambda_3.csv", skiprows=1)
        self.assert_correct_node_sampling_discrete(Poisson(mean=3), pois_samples, False)

        bern_samples = np.loadtxt(fpath + "parametric_samples/bern_prob0.7.csv", skiprows=1)
        self.assert_correct_node_sampling_discrete(Bernoulli(p=0.7), bern_samples, False)

        norm_samples = np.loadtxt(fpath + "parametric_samples/norm_mean10_sd3.csv", skiprows=1)
        self.assert_correct_node_sampling_continuous(Gaussian(mean=10, stdev=3), norm_samples, False)

        gamma_samples = np.loadtxt(fpath + "parametric_samples/gamma_shape2_scale0.5.csv", skiprows=1)
        self.assert_correct_node_sampling_continuous(Gamma(alpha=2, beta=2), gamma_samples, False)

        lognormal_samples = np.loadtxt(fpath + "parametric_samples/lnorm_meanlog_10_sdlog_3.csv", skiprows=1)
        self.assert_correct_node_sampling_continuous(LogNormal(mean=10, stdev=3), lognormal_samples, False)

        exp_samples = np.loadtxt(fpath + "parametric_samples/exp_rate_2.csv", skiprows=1)
        self.assert_correct_node_sampling_continuous(Exponential(l=2), exp_samples, False)


if __name__ == "__main__":
    unittest.main()
