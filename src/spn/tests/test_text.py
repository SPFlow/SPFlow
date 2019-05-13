"""
Created on March 22, 2018

@author: Alejandro Molina
"""
import unittest

from spn.structure.StatisticalTypes import MetaType

from spn.io.Text import str_to_spn, spn_to_str_equation
from spn.structure.Base import Sum, assign_ids, rebuild_scopes_bottom_up, Context
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf

from spn.structure.leaves.parametric.Parametric import *


class TestText(unittest.TestCase):
    def check_obj_and_reconstruction(self, obj):

        str_val = spn_to_str_equation(obj)

        obj_val = str_to_spn(str_val)

        self.assertEqual(str_val, spn_to_str_equation(obj_val))

    def test_json(self):
        # TODO: add test for spn to json
        pass

    def test_histogram_to_str_and_back(self):

        data = np.array([1, 1, 2, 3, 3, 3]).reshape(-1, 1)
        ds_context = Context([MetaType.DISCRETE])
        ds_context.add_domains(data)
        hist = create_histogram_leaf(data, ds_context, [0], alpha=False)

        self.check_obj_and_reconstruction(hist)

    def test_spn_to_str_and_back(self):
        self.check_obj_and_reconstruction(Gaussian(mean=0, stdev=10, scope=0))

        self.check_obj_and_reconstruction(Categorical(p=[0.1, 0.2, 0.7], scope=0))

        self.check_obj_and_reconstruction(Gaussian(mean=0, stdev=10, scope=0))
        self.check_obj_and_reconstruction(Gaussian(mean=1.2, stdev=1.5, scope=0))

        self.check_obj_and_reconstruction(Gaussian(mean=-1.2, stdev=1, scope=0))

        gamma = Gamma(alpha=1, beta=2, scope=0)
        lnorm = LogNormal(mean=1, stdev=2, scope=0)

        self.check_obj_and_reconstruction(gamma)

        self.check_obj_and_reconstruction(lnorm)

        root = Sum(children=[gamma, lnorm], weights=[0.2, 0.8])
        assign_ids(root)
        rebuild_scopes_bottom_up(root)
        self.check_obj_and_reconstruction(root)

        root = 0.3 * (Gaussian(mean=0, stdev=1, scope=0) * Gaussian(mean=1, stdev=1, scope=1)) + 0.7 * (
            Gaussian(mean=2, stdev=1, scope=0) * Gaussian(mean=3, stdev=1, scope=1)
        )

        self.check_obj_and_reconstruction(root)

        # TODO: add test for histograms and pwl


if __name__ == "__main__":
    unittest.main()
