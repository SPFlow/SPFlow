import unittest

from test_base import TestBase
from test_cpp import TestCPP
from test_dsl import TestDSL
from test_expectations import TestParametric
# from test_global_weights
from test_histogram import TestParametric
from test_inference import TestInference
from test_mpe_spn import TestMPE
from test_parametric_mle import TestParametric
from test_parametric import TestParametric
from test_parametric_sampling import TestParametricSampling
from test_piecewise_range import TestPiecewiseRange
from test_posterior import TestPosterior
from test_pwl import TestParametric
# from test_rat_spn
from test_sampling_spn import TestSampling
# from test_tensorflow
from test_transformation import TestTransformation


if __name__ == "__main__":
    print("Running all tests")
    unittest.main()
