"""
Created on Novenber 22, 2018

@author: Zhongjie Yu

"""

import unittest

import numpy as np

from spn.algorithms.measures.WeightOfEvidence import weight_of_evidence, def_w_of_e, conditional_probability
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType


class TestRelevanceScore(unittest.TestCase):
    def test_we_score(self):
        # test if we_score is correct
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
        # test
        n = 40000
        x_instance = np.array([1, 1, 0], dtype=float).reshape(1, -1)
        y_index = 0
        we = weight_of_evidence(spn, 0, x_instance, n, ds_context.domains[y_index].shape[0])
        we_true = np.array([[np.nan, 0, 0]])
        we = we[~np.isnan(we)]
        we_true = we_true[~np.isnan(we_true)]
        self.assertTrue((we == we_true).all())

    def test_def_we(self):
        # test if def_we is correct
        p1 = 0.5
        p2 = 0.5
        we = def_w_of_e(p1, p2)
        self.assertAlmostEqual(we, 0)
        # p2>p1
        p2 = 0.6
        we = def_w_of_e(p1, p2)
        self.assertTrue(we < 0)
        # p2<p1
        p2 = 0.4
        we = def_w_of_e(p1, p2)
        self.assertTrue(we > 0)

    def test_conditional_probability(self):
        # test if conditional probability is correct
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
        # tests
        x_instance = np.array([1, 1, 0], dtype=float).reshape(1, -1)
        self.assertAlmostEqual(conditional_probability(spn, 2, x_instance)[0][0], 0.9)
        self.assertAlmostEqual(conditional_probability(spn, 0, x_instance)[0][0], 0.48)
        x_instance = np.array([2, 1, 0], dtype=float).reshape(1, -1)
        self.assertAlmostEqual(conditional_probability(spn, 0, x_instance)[0][0], 0.36)


if __name__ == "__main__":
    unittest.main()
