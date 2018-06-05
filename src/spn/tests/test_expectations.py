import unittest

from spn.algorithms.stats.Expectations import Expectation
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Expectation import add_histogram_expectation_support
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.parametric.Expectation import add_parametric_expectation_support
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.piecewise.Expectation import add_piecewise_expectation_support
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf


class TestParametric(unittest.TestCase):
    def setUp(self):
        np.random.seed(17)
        add_parametric_inference_support()
        add_parametric_expectation_support()
        add_histogram_expectation_support()
        add_piecewise_expectation_support()

    def test_Parametric_expectations(self):
        spn = 0.3 * (Gaussian(1.0, 1.0, scope=[0]) * Gaussian(5.0, 1.0, scope=[1])) + \
              0.7 * (Gaussian(10.0, 1.0, scope=[0]) * Gaussian(15.0, 1.0, scope=[1]))

        expectation = Expectation(spn, set([0]), None, None)

        self.assertAlmostEqual(0.3 * 1.0 + 0.7 * 10.0, expectation[0, 0], 3)

        expectation = Expectation(spn, set([1]), None, None)
        self.assertAlmostEqual(0.3 * 5.0 + 0.7 * 15.0, expectation[0, 0], 3)

    def test_Histogram_expectations(self):
        data = np.random.randn(20000).reshape(-1, 1)
        ds_context = Context(meta_types=[MetaType.REAL])
        ds_context.add_domains(data)
        hl = create_histogram_leaf(data, ds_context, scope=[0])
        expectation = Expectation(hl, set([0]), None, None)

        self.assertAlmostEqual(np.mean(data[:, 0]), expectation[0, 0], 3)

        data = np.random.randint(0, high=100, size=20000).reshape(-1, 1)
        ds_context = Context(meta_types=[MetaType.DISCRETE])
        ds_context.add_domains(data)
        hl = create_histogram_leaf(data, ds_context, scope=[0])
        expectation = Expectation(hl, set([0]), None, None)

        self.assertAlmostEqual(np.mean(data[:, 0]), expectation[0, 0], 3)

    def test_Piecewise_expectations(self):
        data = np.random.normal(loc=100.0, scale=5.00, size=20000).reshape(-1, 1)
        ds_context = Context(meta_types=[MetaType.REAL])
        ds_context.add_domains(data)
        pl = create_piecewise_leaf(data, ds_context, scope=[0], prior_weight=None)
        expectation = Expectation(pl, set([0]), None, None)

        self.assertAlmostEqual(np.mean(data[:, 0]), expectation[0, 0], 2)

        data = np.random.randint(0, high=100, size=20000).reshape(-1, 1)
        ds_context = Context(meta_types=[MetaType.DISCRETE])
        ds_context.add_domains(data)
        pl = create_piecewise_leaf(data, ds_context, scope=[0], prior_weight=None)
        expectation = Expectation(pl, set([0]), None, None)

        self.assertAlmostEqual(np.mean(data[:, 0]), expectation[0, 0], 3)

    def test_Piecewise_expectations_with_evidence(self):
        adata = np.zeros((20000, 2))
        adata[:, 1] = 0
        adata[:, 0] = np.random.normal(loc=100.0, scale=5.00, size=adata.shape[0])

        bdata = np.zeros_like(adata)
        bdata[:, 1] = 1
        bdata[:, 0] = np.random.normal(loc=50.0, scale=5.00, size=bdata.shape[0])

        data = np.vstack((adata, bdata))

        ds_context = Context(meta_types=[MetaType.REAL, MetaType.DISCRETE])
        ds_context.parametric_type = [None, Categorical]
        ds_context.add_domains(data)
        L = (create_piecewise_leaf(adata[:, 0].reshape(-1, 1), ds_context, scope=[0], prior_weight=None) *
             create_parametric_leaf(adata[:, 1].reshape(-1, 1), ds_context, scope=[1]))
        R = (create_piecewise_leaf(bdata[:, 0].reshape(-1, 1), ds_context, scope=[0], prior_weight=None) *
             create_parametric_leaf(bdata[:, 1].reshape(-1, 1), ds_context, scope=[1]))

        spn = 0.5 * L + 0.5 * R

        evidence = np.zeros((2, 2))
        evidence[1, 1] = 1
        expectation = Expectation(spn, set([0]), set([1]), evidence)

        self.assertAlmostEqual(np.mean(adata[:, 0]), expectation[0, 0], 2)
        self.assertAlmostEqual(np.mean(bdata[:, 0]), expectation[1, 0], 2)


if __name__ == '__main__':
    unittest.main()
