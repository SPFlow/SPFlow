import unittest

import numpy as np
import pytest

from spn.algorithms.Inference import log_likelihood
from spn.io.CPP import get_cpp_function, setup_cpp_bridge
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import Gaussian

try:
    import cppyy
    CPPYY_AVAILABLE = True
except ImportError:
    CPPYY_AVAILABLE = False


@pytest.mark.skipif(not CPPYY_AVAILABLE, reason="cppyy not installed")
class TestCPP(unittest.TestCase):
    def setUp(self):
        add_parametric_inference_support()

    def test_bcpp(self):
        D = Gaussian(mean=1.0, stdev=1.0, scope=[0])
        E = Gaussian(mean=2.0, stdev=2.0, scope=[1])
        F = Gaussian(mean=3.0, stdev=3.0, scope=[0])
        G = Gaussian(mean=4.0, stdev=4.0, scope=[1])

        B = D * E
        C = F * G

        A = 0.3 * B + 0.7 * C

        setup_cpp_bridge(A)
        spn_cc_eval_func = get_cpp_function(A)

        np.random.seed(17)
        data = np.random.normal(10, 0.01, size=200000).tolist() + np.random.normal(30, 10, size=200000).tolist()
        data = np.array(data).reshape((-1, 2))

        py_ll = log_likelihood(A, data)

        c_ll = spn_cc_eval_func(data)

        for i in range(py_ll.shape[0]):
            self.assertAlmostEqual(py_ll[i, 0], c_ll[i, 0])


if __name__ == "__main__":
    unittest.main()
