import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isclose, tl_vstack, tl_isnan, tl_allclose

from spflow.tensorly.learning import maximum_likelihood_estimation
from spflow.tensorly.structure.spn import MultivariateGaussian
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_mle_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = MultivariateGaussian(Scope([0, 1]))

        # simulate data
        data = np.random.multivariate_normal(
            mean=tl.tensor([-1.7, 0.3]),
            cov=tl.tensor([[1.0, 0.25], [0.25, 0.5]]),
            size=(10000,),
        )

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(tl_allclose(leaf.mean, tl.tensor([-1.7, 0.3]), atol=1e-2, rtol=1e-2))
        self.assertTrue(
            tl_allclose(
                leaf.cov,
                tl.tensor([[1.0, 0.25], [0.25, 0.5]]),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    def test_mle_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        leaf = MultivariateGaussian(Scope([0, 1]))

        # simulate data
        data = np.random.multivariate_normal(
            mean=tl.tensor([0.5, 0.2]),
            cov=tl.tensor([[1.3, -0.7], [-0.7, 1.0]]),
            size=(10000,),
        )

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(tl_allclose(leaf.mean, tl.tensor([0.5, 0.2]), atol=1e-2, rtol=1e-2))
        self.assertTrue(
            tl_allclose(
                leaf.cov,
                tl.tensor([[1.3, -0.7], [-0.7, 1.0]]),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    def test_mle_bias_correction(self):

        leaf = MultivariateGaussian(Scope([0, 1]))
        data = tl.tensor([[-1.0, 1.0], [1.0, 0.5]])

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=False)
        self.assertTrue(tl_allclose(leaf.cov, tl.tensor([[1.0, -0.25], [-0.25, 0.0625]])))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)
        self.assertTrue(tl_allclose(leaf.cov, 2 * tl.tensor([[1.0, -0.25], [-0.25, 0.0625]])))

    def test_mle_edge_cov_zero(self):

        leaf = MultivariateGaussian(Scope([0, 1]))

        # simulate data
        data = tl.tensor([[-1.0, 1.0]])

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=False)
        # without bias correction diagonal values are zero and should be set to larger value
        self.assertTrue(tl.all(tl.diag(leaf.cov) > 0))

    def test_mle_only_nans(self):

        leaf = MultivariateGaussian(Scope([0, 1]))

        # simulate data
        data = tl.tensor([[tl.nan, tl.nan], [tl.nan, tl.nan]])

        # check if exception is raised
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            data,
            nan_strategy="ignore",
        )

    def test_mle_invalid_support(self):

        leaf = MultivariateGaussian(Scope([0, 1]))

        # perform MLE (should raise exceptions)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            tl.tensor([[tl.inf, 0.0]]),
            bias_correction=True,
        )

    def test_mle_nan_strategy_none(self):

        leaf = MultivariateGaussian(Scope([0, 1]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            tl.tensor([[tl.nan, 0.0], [-2.3, 0.1], [-1.8, 1.9], [0.9, 0.7]]),
            nan_strategy=None,
        )

    def test_mle_nan_strategy_ignore(self):

        leaf = MultivariateGaussian(Scope([0, 1]))
        # row of NaN values since partially missing rows are not taken into account by numpy.ma.cov and therefore results in different result
        maximum_likelihood_estimation(
            leaf,
            tl.exp(tl.tensor([[tl.nan, tl.nan], [-2.3, 0.1], [-1.8, 1.9], [0.9, 0.7]])),
            nan_strategy="ignore",
            bias_correction=False,
        )
        mean_ignore, cov_ignore = leaf.mean, leaf.cov

        maximum_likelihood_estimation(
            leaf,
            tl.exp(tl.tensor([[-2.3, 0.1], [-1.8, 1.9], [0.9, 0.7]])),
            nan_strategy=None,
            bias_correction=False,
        )
        mean_none, cov_none = leaf.mean, leaf.cov

        self.assertTrue(tl_allclose(mean_ignore, mean_none))
        self.assertTrue(tl_allclose(cov_ignore, cov_none))

    def test_mle_nan_strategy_callable(self):

        leaf = MultivariateGaussian(Scope([0, 1]))
        # should not raise an issue
        maximum_likelihood_estimation(leaf, tl.tensor([[0.5, 1.0], [-1.0, 0.0]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        leaf = MultivariateGaussian(Scope([0, 1]))
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            tl.tensor([[tl.nan, 0.0], [1, 0.1], [1.9, -0.2]]),
            nan_strategy="invalid_string",
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            tl.tensor([[tl.nan, 0.0], [1, 0.1], [1.9, -0.2]]),
            nan_strategy=1,
        )

    def test_weighted_mle(self):

        leaf = MultivariateGaussian(Scope([0, 1]))

        data = tl_vstack(
            [
                np.random.multivariate_normal([1.7, 2.1], tl.eye(2), size=(10000,)),
                np.random.multivariate_normal([0.5, -0.3], tl.eye(2), size=(10000,)),
            ]
        )
        weights = tl.concatenate([tl.zeros(10000), tl.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(tl_allclose(leaf.mean, tl.tensor([0.5, -0.3]), atol=1e-2, rtol=1e-1))
        self.assertTrue(tl_allclose(leaf.cov, tl.eye(2), atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
