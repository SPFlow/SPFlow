import unittest

from spn.algorithms.EM import EM_optimization
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
import numpy as np


from spn.structure.leaves.parametric.Parametric import Gaussian


class TestEM(unittest.TestCase):
    def test_optimization(self):
        np.random.seed(17)
        d1 = np.random.normal(10, 5, size=2000).tolist()
        d2 = np.random.normal(30, 5, size=2000).tolist()
        data = d1 + d2
        data = np.array(data).reshape((-1, 10))
        data = data.astype(np.float32)

        ds_context = Context(meta_types=[MetaType.REAL] * data.shape[1], parametric_types=[Gaussian] * data.shape[1])

        spn = learn_parametric(data, ds_context)

        spn.weights = [0.8, 0.2]
        spn.children[0].children[0].mean = 3.0

        py_ll = np.sum(log_likelihood(spn, data))

        print(spn.weights, spn.children[0].children[0].mean)

        EM_optimization(spn, data, iterations=10)

        print(spn.weights, spn.children[0].children[0].mean)

        py_ll_opt = np.sum(log_likelihood(spn, data))

        self.assertLessEqual(py_ll, py_ll_opt)
        self.assertAlmostEqual(spn.weights[0], 0.5)
        self.assertAlmostEqual(spn.weights[1], 0.5)
        self.assertAlmostEqual(spn.children[0].children[0].mean, 10.50531, 4)


if __name__ == "__main__":
    unittest.main()
