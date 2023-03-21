import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose, tl_vstack

from spflow.tensorly.learning import maximum_likelihood_estimation
from spflow.tensorly.structure.spn import MultivariateGaussianLayer
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_mle(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        layer = MultivariateGaussianLayer(scope=[Scope([0, 1]), Scope([2, 3])])

        # simulate data
        data = np.hstack(
            [
                np.random.multivariate_normal(
                    mean=tl.tensor([-1.7, 0.3]),
                    cov=tl.tensor([[1.0, 0.25], [0.25, 0.5]]),
                    size=(20000,),
                ),
                np.random.multivariate_normal(
                    mean=tl.tensor([0.5, 0.2]),
                    cov=tl.tensor([[1.3, -0.7], [-0.7, 1.0]]),
                    size=(20000,),
                ),
            ]
        )

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(
            tl_allclose(
                layer.mean,
                tl.tensor([[-1.7, 0.3], [0.5, 0.2]]),
                atol=1e-2,
                rtol=1e-2,
            )
        )
        self.assertTrue(
            tl_allclose(
                layer.cov,
                tl.tensor([[[1.0, 0.25], [0.25, 0.5]], [[1.3, -0.7], [-0.7, 1.0]]]),
                atol=1e-2,
                rtol=1e-2,
            )
        )


if __name__ == "__main__":
    unittest.main()
