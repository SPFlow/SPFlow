"""
Created on Novenber 22, 2018

@author: Zhongjie Yu

"""

import unittest

from spn.algorithms.measures.InformationTheory import *
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType


class TestMutualInfo(unittest.TestCase):
    def test_entropy(self):
        # test if entropy is correct
        """
        # explain how training data and the spn comes
        # number of RVs
        M = 3
        # table of probabilities
        p1 = 0.6
        p2 = 0.3
        p31 = 0.1
        p32 = 0.9
        # generate x1 and x2
        x1 = np.random.binomial(1, p1, size=N) + np.random.binomial(1, p1, size=N)
        x2 = np.random.binomial(1, p2, size=N)
        x3 = np.zeros(N)
        # generate x3
        for i in range(N):
            if x2[i] == 1:
                x3[i] = np.random.binomial(1, p31, size=1)
            else:
                x3[i] = np.random.binomial(1, p32, size=1)
        # form a matrix, rows are instances and columns are RVs
        train_data = np.concatenate((x1, x2, x3)).reshape((M, N)).transpose()
        """
        # only for generating the ds_context
        train_data = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
        # spn
        ds_context = Context(meta_types=[MetaType.DISCRETE] * 3)
        ds_context.add_domains(train_data)
        ds_context.parametric_type = [Categorical] * 3
        spn = 0.64 * (
            (
                Categorical(p=[0.25, 0.75, 0.0], scope=0)
                * (
                    0.34 * ((Categorical(p=[7 / 34, 27 / 34], scope=1) * Categorical(p=[1.0, 0.0], scope=2)))
                    + 0.66 * ((Categorical(p=[21 / 22, 1 / 22], scope=1) * Categorical(p=[0.0, 1.0], scope=2)))
                )
            )
        ) + 0.36 * (
            (
                Categorical(p=[0.0, 0.0, 1.0], scope=0)
                * (
                    0.34 * ((Categorical(p=[7 / 34, 27 / 34], scope=1) * Categorical(p=[1.0, 0.0], scope=2)))
                    + 0.66 * ((Categorical(p=[21 / 22, 1 / 22], scope=1) * Categorical(p=[0.0, 1.0], scope=2)))
                )
            )
        )
        # real entropy
        p2 = 0.3
        h_x2 = -p2 * np.log(p2) - (1 - p2) * np.log(1 - p2)
        self.assertAlmostEqual(h_x2, entropy(spn, ds_context, {1}))
        h_x2x3 = -(p2 * np.log(p2) + (1 - p2) * np.log(1 - p2) + 0.9 * np.log(0.9) + 0.1 * np.log(0.1))
        self.assertAlmostEqual(h_x2x3, entropy(spn, ds_context, {1, 2}))
        h_x1 = -(0.16 * np.log(0.16) + 0.36 * np.log(0.36) + 0.48 * np.log(0.48))
        self.assertAlmostEqual(h_x1, entropy(spn, ds_context, {0}))
        h_x2x1 = -(0.7 * np.log(0.7) + 0.3 * np.log(0.3)) + h_x1
        self.assertAlmostEqual(h_x2x1, entropy(spn, ds_context, {1, 0}))
        h_x3x1 = -(0.66 * np.log(0.66) + 0.34 * np.log(0.34)) + h_x1
        self.assertAlmostEqual(h_x3x1, entropy(spn, ds_context, {2, 0}))
        h_x2x3x1 = h_x1 + h_x2x3
        self.assertAlmostEqual(h_x2x3x1, entropy(spn, ds_context, {1, 2, 0}))
        # test symmetry
        self.assertAlmostEqual(entropy(spn, ds_context, {0, 2}), entropy(spn, ds_context, {2, 0}))
        self.assertAlmostEqual(entropy(spn, ds_context, {1, 2}), entropy(spn, ds_context, {2, 1}))

    def test_mutual_info(self):
        # test if mutual info is correct
        # same spn as in entropy test
        # only for generating the ds_context
        train_data = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
        # spn
        ds_context = Context(meta_types=[MetaType.DISCRETE] * 3)
        ds_context.add_domains(train_data)
        ds_context.parametric_type = [Categorical] * 3
        spn = 0.64 * (
            (
                Categorical(p=[0.25, 0.75, 0.0], scope=0)
                * (
                    0.34 * ((Categorical(p=[7 / 34, 27 / 34], scope=1) * Categorical(p=[1.0, 0.0], scope=2)))
                    + 0.66 * ((Categorical(p=[21 / 22, 1 / 22], scope=1) * Categorical(p=[0.0, 1.0], scope=2)))
                )
            )
        ) + 0.36 * (
            (
                Categorical(p=[0.0, 0.0, 1.0], scope=0)
                * (
                    0.34 * ((Categorical(p=[7 / 34, 27 / 34], scope=1) * Categorical(p=[1.0, 0.0], scope=2)))
                    + 0.66 * ((Categorical(p=[21 / 22, 1 / 22], scope=1) * Categorical(p=[0.0, 1.0], scope=2)))
                )
            )
        )
        # real mutual info
        p2 = 0.3
        p3 = 0.66
        h_x2 = -p2 * np.log(p2) - (1 - p2) * np.log(1 - p2)
        h_x3 = -p3 * np.log(p3) - (1 - p3) * np.log(1 - p3)
        h_x2x3 = -(p2 * np.log(p2) + (1 - p2) * np.log(1 - p2) + 0.9 * np.log(0.9) + 0.1 * np.log(0.1))
        mi_x2x3 = h_x2 + h_x3 - h_x2x3
        self.assertAlmostEqual(mi_x2x3, mutual_information(spn, ds_context, {1}, {2}))
        mi_x1x2 = 0
        self.assertAlmostEqual(mi_x1x2, mutual_information(spn, ds_context, {1}, {0}))
        # test symmetry
        self.assertAlmostEqual(
            mutual_information(spn, ds_context, {2}, {1}), mutual_information(spn, ds_context, {1}, {2})
        )
        self.assertAlmostEqual(
            mutual_information(spn, ds_context, {0, 2}, {1}), mutual_information(spn, ds_context, {1}, {0, 2})
        )
        # rest 0
        self.assertAlmostEqual(0, mutual_information(spn, ds_context, {2, 1}, {0}))

    def test_conditional_mutual_info(self):
        # test if conditional mutual info is correct
        # same spn as in entropy test
        # only for generating the ds_context
        train_data = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
        # spn
        ds_context = Context(meta_types=[MetaType.DISCRETE] * 3)
        ds_context.add_domains(train_data)
        ds_context.parametric_type = [Categorical] * 3
        spn = 0.64 * (
            (
                Categorical(p=[0.25, 0.75, 0.0], scope=0)
                * (
                    0.34 * ((Categorical(p=[7 / 34, 27 / 34], scope=1) * Categorical(p=[1.0, 0.0], scope=2)))
                    + 0.66 * ((Categorical(p=[21 / 22, 1 / 22], scope=1) * Categorical(p=[0.0, 1.0], scope=2)))
                )
            )
        ) + 0.36 * (
            (
                Categorical(p=[0.0, 0.0, 1.0], scope=0)
                * (
                    0.34 * ((Categorical(p=[7 / 34, 27 / 34], scope=1) * Categorical(p=[1.0, 0.0], scope=2)))
                    + 0.66 * ((Categorical(p=[21 / 22, 1 / 22], scope=1) * Categorical(p=[0.0, 1.0], scope=2)))
                )
            )
        )
        # real mutual info
        p2 = 0.3
        p3 = 0.66
        h_x1 = -(0.16 * np.log(0.16) + 0.36 * np.log(0.36) + 0.48 * np.log(0.48))
        h_x2x1 = -(0.7 * np.log(0.7) + 0.3 * np.log(0.3)) + h_x1
        h_x3x1 = -(0.66 * np.log(0.66) + 0.34 * np.log(0.34)) + h_x1
        h_x2x3 = -(p2 * np.log(p2) + (1 - p2) * np.log(1 - p2) + 0.9 * np.log(0.9) + 0.1 * np.log(0.1))
        h_x2x3x1 = h_x1 + h_x2x3
        cmi_x2x3_x1 = h_x2x1 + h_x3x1 - h_x2x3x1 - h_x1
        self.assertAlmostEqual(cmi_x2x3_x1, conditional_mutual_information(spn, ds_context, {1}, {2}, {0}))
        h_x1x3 = h_x3x1
        h_x1x2x3 = h_x2x3x1
        h_x3 = -p3 * np.log(p3) - (1 - p3) * np.log(1 - p3)
        cmi_x1x2_x3 = h_x1x3 + h_x2x3 - h_x1x2x3 - h_x3
        self.assertAlmostEqual(cmi_x1x2_x3, conditional_mutual_information(spn, ds_context, {1}, {0}, {2}))
        h_x1x2x3 = h_x2x3x1
        h_x2 = -p2 * np.log(p2) - (1 - p2) * np.log(1 - p2)
        cmi_x1x3_x2 = h_x2x1 + h_x2x3 - h_x1x2x3 - h_x2
        self.assertAlmostEqual(cmi_x1x3_x2, conditional_mutual_information(spn, ds_context, {2}, {0}, {1}))


if __name__ == "__main__":
    unittest.main()
