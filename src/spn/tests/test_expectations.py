import unittest

from spn.algorithms.stats.Expectations import Expectation
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf


class TestParametric(unittest.TestCase):
    def setUp(self):
        np.random.seed(17)

    def test_Parametric_expectations(self):
        spn = 0.3 * (Gaussian(1.0, 1.0, scope=[0]) * Gaussian(5.0, 1.0, scope=[1])) + 0.7 * (
            Gaussian(10.0, 1.0, scope=[0]) * Gaussian(15.0, 1.0, scope=[1])
        )

        expectation = Expectation(spn, set([0]))

        self.assertAlmostEqual(0.3 * 1.0 + 0.7 * 10.0, expectation[0, 0], 3)

        expectation = Expectation(spn, set([1]))
        self.assertAlmostEqual(0.3 * 5.0 + 0.7 * 15.0, expectation[0, 0], 3)

    def test_Histogram_expectations(self):
        data = np.random.randn(20000).reshape(-1, 1)
        ds_context = Context(meta_types=[MetaType.REAL])
        ds_context.add_domains(data)
        hl = create_histogram_leaf(data, ds_context, scope=[0])
        expectation = Expectation(hl, set([0]))

        self.assertAlmostEqual(np.mean(data[:, 0]), expectation[0, 0], 3)

        data = np.random.randint(0, high=100, size=20000).reshape(-1, 1)
        ds_context = Context(meta_types=[MetaType.DISCRETE])
        ds_context.add_domains(data)
        hl = create_histogram_leaf(data, ds_context, scope=[0])
        expectation = Expectation(hl, set([0]))

        self.assertAlmostEqual(np.mean(data[:, 0]), expectation[0, 0], 3)

    def test_Piecewise_expectations(self):
        data = np.random.normal(loc=100.0, scale=5.00, size=20000).reshape(-1, 1)
        ds_context = Context(meta_types=[MetaType.REAL])
        ds_context.add_domains(data)
        pl = create_piecewise_leaf(data, ds_context, scope=[0], prior_weight=None)
        expectation = Expectation(pl, set([0]))

        self.assertAlmostEqual(np.mean(data[:, 0]), expectation[0, 0], 2)

        data = np.random.randint(0, high=100, size=2000).reshape(-1, 1)
        ds_context = Context(meta_types=[MetaType.DISCRETE])
        ds_context.add_domains(data)
        pl = create_piecewise_leaf(data, ds_context, scope=[0], prior_weight=None)
        expectation = Expectation(pl, set([0]))

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
        ds_context.parametric_types = [None, Categorical]
        ds_context.add_domains(data)
        L = create_piecewise_leaf(
            adata[:, 0].reshape(-1, 1), ds_context, scope=[0], prior_weight=None, hist_source="numpy"
        ) * create_parametric_leaf(adata[:, 1].reshape(-1, 1), ds_context, scope=[1])
        R = create_piecewise_leaf(
            bdata[:, 0].reshape(-1, 1), ds_context, scope=[0], prior_weight=None, hist_source="numpy"
        ) * create_parametric_leaf(bdata[:, 1].reshape(-1, 1), ds_context, scope=[1])

        spn = 0.5 * L + 0.5 * R

        evidence = np.zeros((2, 2))
        evidence[1, 1] = 1
        evidence[:, 0] = np.nan
        expectation = Expectation(spn, set([0]), evidence)

        self.assertAlmostEqual(np.mean(adata[:, 0]), expectation[0, 0], 2)
        self.assertAlmostEqual(np.mean(bdata[:, 0]), expectation[1, 0], 2)

    def test_Piecewise_full(self):
        from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear

        # In order to create piecewise nodes
        def create_piecewise_node(x_range, y_range, scope):
            x_range, y_range = np.array(x_range), np.array(y_range)
            auc = np.trapz(y_range, x_range)
            y_range = y_range / auc
            return PiecewiseLinear(x_range=x_range, y_range=y_range, bin_repr_points=x_range[1:-1], scope=scope)

        # Create node
        node1 = create_piecewise_node([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 10.0, 20.0, 30.0, 0.0], [0])

        node2 = create_piecewise_node([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 30.0, 20.0, 10.0, 0.0], [0])

        # Test expectation node1
        true_value = 2.33333333
        expectation = Expectation(node1, set([0]))
        self.assertAlmostEqual(true_value, expectation[0, 0], 5)

        # Test expectation node2
        true_value = 1.66666666
        expectation = Expectation(node2, set([0]))
        self.assertAlmostEqual(true_value, expectation[0, 0], 5)

        # Test expectation with evidence
        true_value = 1.0
        evidence = np.zeros((1, 1))
        evidence[0, 0] = 1.0
        with self.assertRaises(AssertionError):
            expectation = Expectation(node2, set([0]), evidence)
            # self.assertAlmostEqual(true_value, expectation[0, 0], 5)
            """
            Above fails because the evidence is ignored on features for which the expectation
            is computed. This is even more important if we evaluate ranges.
            """

        # Test expectation of SPN with node1 and node2
        spn1 = 0.5 * node1 + 0.5 * node2
        true_value = 2.0
        expectation = Expectation(spn1, set([0]))
        self.assertAlmostEqual(true_value, expectation[0, 0], 5)

        # Create more nodes
        node3 = create_piecewise_node([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 10.0, 10.0, 10.0, 0.0, 0.0], [1])

        node4 = create_piecewise_node([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 10.0, 10.0, 10.0, 0.0], [1])

        # Test expectation node3
        true_value = 2.0
        expectation = Expectation(node3, set([1]))
        self.assertAlmostEqual(true_value, expectation[0, 0], 5)

        # Test expectation node4
        true_value = 3.0
        expectation = Expectation(node4, set([1]))
        self.assertAlmostEqual(true_value, expectation[0, 0], 5)

        # Test expectation of SPN with node1, node2, node3 and node4
        spn2 = 0.5 * (node1 * node3) + 0.5 * (node2 * node4)
        true_value = 2.5
        expectation = Expectation(spn2, set([1]))
        self.assertAlmostEqual(true_value, expectation[0, 0], 5)

        # Probability of both subtrees is the same due to the evidence
        # since the expectation of node3 and node3 have the same weight
        # resulting in expectation of 2.5
        true_value = 2.5
        evidence = np.zeros((1, 2))
        evidence[0, 0] = 2.0  # Since node1 and node2 return 33% the true value will be the same as without evidence
        evidence[0, 1] = np.nan
        expectation = Expectation(spn2, set([1]), evidence)
        self.assertAlmostEqual(true_value, expectation[0, 0], 5)

        # Probability of right subtree will be higher due to the evidence
        # since node2 has a higher probability for 1. than node1
        # Hence the expectation of node4 has a higher impact
        evidence = np.zeros((1, 2))
        evidence[0, 0] = 1.0
        evidence[0, 1] = np.nan
        expectation = Expectation(spn2, set([1]), evidence)
        self.assertTrue(2.5 < expectation[0, 0], 5)

        # Probability of left subtree will be higher due to the evidence
        # since node1 has a higher probability for 3. than node2
        # Hence the expectation of node3 has a higher impact
        evidence = np.zeros((1, 2))
        evidence[0, 0] = 3.0
        evidence[0, 1] = np.nan
        expectation = Expectation(spn2, set([1]), evidence)
        self.assertTrue(2.5 > expectation[0, 0], 5)

        # this test does not conform to the expected behavior of the spn
        # with self.assertRaises(AssertionError):
        #     # Test with evidence is None
        #     expectation = Expectation(spn2, set([0]), set([1]), None)
        #     '''
        #     Above fails because the the fake evidence which is defined in
        #     spn.algorithms.stats.Expectation
        #     is only of column-length 1 but we have 2 features and access the
        #     second feature in the likelihood-method
        #     '''


if __name__ == "__main__":
    unittest.main()
