import unittest

import numpy as np
import pytest

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.MPE import mpe
from spn.io.CPP import get_cpp_function, setup_cpp_bridge, get_cpp_mpe_function
from spn.io.Graphics import plot_spn
from spn.structure.Base import get_nodes_by_type
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import Gaussian, Bernoulli

try:
    import cppyy
    CPPYY_AVAILABLE = True
except ImportError:
    CPPYY_AVAILABLE = False


@pytest.mark.skipif(not CPPYY_AVAILABLE, reason="cppyy not installed")
class TestCPP(unittest.TestCase):
    def setUp(self):
        add_parametric_inference_support()

    def test_binary(self):

        A = 0.4 * (
            Bernoulli(p=0.8, scope=0)
            * (
                0.3 * (Bernoulli(p=0.7, scope=1) * Bernoulli(p=0.6, scope=2))
                + 0.7 * (Bernoulli(p=0.5, scope=1) * Bernoulli(p=0.4, scope=2))
            )
        ) + 0.6 * (Bernoulli(p=0.8, scope=0) * Bernoulli(p=0.7, scope=1) * Bernoulli(p=0.6, scope=2))

        setup_cpp_bridge(A)

        spn_cc_eval_func_bernoulli = get_cpp_function(A)
        num_data = 200000

        data = (
            np.random.binomial(1, 0.3, size=(num_data)).astype("float32").tolist()
            + np.random.binomial(1, 0.3, size=(num_data)).astype("float32").tolist()
            + np.random.binomial(1, 0.3, size=(num_data)).astype("float32").tolist()
        )

        data = np.array(data).reshape((-1, 3))

        num_nodes = len(get_nodes_by_type(A))

        lls_matrix = np.zeros((num_data, num_nodes))

        # Test for every single lls_maxtrix element.
        _ = log_likelihood(A, data, lls_matrix=lls_matrix)
        c_ll = spn_cc_eval_func_bernoulli(data)
        self.assertTrue(np.allclose(lls_matrix, c_ll))

        ### Testing for MPE.
        spn_cc_mpe_func_bernoulli = get_cpp_mpe_function(A)

        # drop some data.
        for i in range(data.shape[0]):
            drop_data = np.random.binomial(data.shape[1] - 1, 0.5)
            data[i, drop_data] = np.nan

        cc_completion = spn_cc_mpe_func_bernoulli(data)
        py_completion = mpe(A, data)
        self.assertTrue(np.allclose(py_completion, cc_completion))


if __name__ == "__main__":

    unittest.main()
