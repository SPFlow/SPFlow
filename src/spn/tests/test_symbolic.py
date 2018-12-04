"""
Created on November 23, 2018

@author: Alejandro Molina
"""
import unittest

from spn.algorithms.Statistics import get_structure_stats

from spn.data.datasets import get_binary_data

from spn.algorithms.Inference import log_likelihood

from spn.structure.leaves.parametric.Parametric import Gaussian, Bernoulli
from spn.structure.Base import Context
from spn.io.Symbolic import spn_to_sympy
import numpy as np


class TestSymbolic(unittest.TestCase):
    def test_bernoulli_spn_ll(self):
        train_data = get_binary_data("dna")[3]
        train_data = train_data[:, 0:3]
        ds_context = Context(parametric_types=[Bernoulli] * 3, feature_names=["x0", "x1", "x2"]).add_domains(train_data)

        from spn.algorithms.LearningWrappers import learn_parametric

        spn = learn_parametric(train_data, ds_context, min_instances_slice=1500)

        print(get_structure_stats(spn))

        sympyecc = spn_to_sympy(spn)

        print(sympyecc)

        # plot_spn(spn, context=ds_context, file_name="/tmp/test_spn.png")

    def test_gaussian_spn_ll(self):
        root = 0.3 * (Gaussian(mean=0, stdev=1, scope=0) * Gaussian(mean=1, stdev=1, scope=1)) + 0.7 * (
            Gaussian(mean=2, stdev=1, scope=0) * Gaussian(mean=3, stdev=1, scope=1)
        )

        sympyecc = spn_to_sympy(root)
        logsympyecc = spn_to_sympy(root, log=True)

        sym_l = float(sympyecc.evalf(subs={"x0": 0, "x1": 0}))
        sym_ll = float(logsympyecc.evalf(subs={"x0": 0, "x1": 0}))

        data = np.array([0, 0], dtype=np.float).reshape(-1, 2)

        self.assertTrue(np.alltrue(np.isclose(np.log(sym_l), log_likelihood(root, data))))
        self.assertTrue(np.alltrue(np.isclose(sym_ll, log_likelihood(root, data))))


if __name__ == "__main__":
    unittest.main()

    spn_to_sympy(root)
