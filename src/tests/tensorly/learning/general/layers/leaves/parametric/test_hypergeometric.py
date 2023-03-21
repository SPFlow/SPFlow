import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose, tl_vstack

from spflow.tensorly.learning import maximum_likelihood_estimation
from spflow.tensorly.structure.spn import HypergeometricLayer
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_mle(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[10, 4], M=[7, 1], n=[5, 3])

        # simulate data
        data = np.hstack(
            [
                np.random.hypergeometric(ngood=7, nbad=10 - 7, nsample=5, size=(10000, 1)),
                np.random.hypergeometric(ngood=1, nbad=4 - 1, nsample=3, size=(10000, 1)),
            ]
        )

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(tl.all(layer.N == tl.tensor([10, 4])))
        self.assertTrue(tl.all(layer.M == tl.tensor([7, 1])))
        self.assertTrue(tl.all(layer.n == tl.tensor([5, 3])))


if __name__ == "__main__":
    unittest.main()
