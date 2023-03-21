import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose, tl_vstack


from spflow.tensorly.learning import maximum_likelihood_estimation
from spflow.tensorly.structure.spn import UniformLayer
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_mle(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        layer = UniformLayer(scope=[Scope([0]), Scope([1])], start=[-3.0, 0.0], end=[-1.0, 1.0])

        # simulate data
        data = np.hstack(
            [
                np.random.uniform(low=-3.0, high=-1.0, size=(100, 1)),
                np.random.uniform(low=0.0, high=1.0, size=(100, 1)),
            ]
        )

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(tl.all(layer.start == tl.tensor([-3.0, 0.0])))
        self.assertTrue(tl.all(layer.end == tl.tensor([-1.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
