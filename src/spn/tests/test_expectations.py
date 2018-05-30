import unittest

from spn.algorithms.Inference import likelihood, log_likelihood
from spn.algorithms.stats.Expectations import Expectation
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Expectation import add_parametric_expectation_support
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import *


class TestParametric(unittest.TestCase):
    def setUp(self):
        add_parametric_inference_support()
        add_parametric_expectation_support()
        self.tested = set()

    def test_Parametric_expectations(self):
        spn = 0.3 * (Gaussian(1.0, 1.0, scope=[0]) * Gaussian(5.0, 1.0, scope=[1])) + \
              0.7 * (Gaussian(10.0, 1.0, scope=[0]) * Gaussian(15.0, 1.0, scope=[1]))

        evidence = np.random.rand(20).reshape(-1, 2)
        ds_context = Context(meta_types=[MetaType.REAL, MetaType.REAL])
        ds_context.add_domains(evidence)

        expectation = Expectation(spn, set([0]), None, evidence, ds_context)

        print("DISCRETE", "mean should be ", np.mean(evidence), "is", expectation)

        for child in Parametric.__subclasses__():
            if child not in self.tested:
                print("not tested", child)

    def test_Parametric_constructors_from_data(self):
        pass


if __name__ == '__main__':
    unittest.main()
