from spflow.meta.data import Scope
from spflow.base.structure.spn import (
    MultivariateGaussianLayer,
)
from spflow.base.learning import (
    maximum_likelihood_estimation,
)

import numpy as np
import unittest
import random


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
                    mean=np.array([-1.7, 0.3]),
                    cov=np.array([[1.0, 0.25], [0.25, 0.5]]),
                    size=(20000,),
                ),
                np.random.multivariate_normal(
                    mean=np.array([0.5, 0.2]),
                    cov=np.array([[1.3, -0.7], [-0.7, 1.0]]),
                    size=(20000,),
                ),
            ]
        )

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(
            np.allclose(
                layer.mean,
                np.array([[-1.7, 0.3], [0.5, 0.2]]),
                atol=1e-2,
                rtol=1e-2,
            )
        )
        self.assertTrue(
            np.allclose(
                layer.cov,
                np.array(
                    [[[1.0, 0.25], [0.25, 0.5]], [[1.3, -0.7], [-0.7, 1.0]]]
                ),
                atol=1e-2,
                rtol=1e-2,
            )
        )


if __name__ == "__main__":
    unittest.main()
