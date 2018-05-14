import unittest

from spn.io.Text import to_JSON, spn_to_str_equation
from spn.structure.Base import assign_ids
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.leaves.parametric.Text import add_parametric_text_support


class TestDSL(unittest.TestCase):
    def setUp(self):
        add_parametric_text_support()

    def test_sum(self):
        spn = 0.5 * Gaussian(0.0, 1.0, scope=0) + 0.5 * Gaussian(2.0, 1.0, scope=0)

        spn_text = "(0.5*(Gaussian(V0|mean=0.0;stdev=1.0)) + 0.5*(Gaussian(V0|mean=2.0;stdev=1.0)))"

        self.assertEqual(spn_to_str_equation(spn), spn_text)

    def test_multiple_sum(self):
        spn = 0.6*(0.4 * Gaussian(0.0, 1.0, scope=0) + 0.6 * Gaussian(2.0, 1.0, scope=0)) + 0.4 * Gaussian(2.0, 1.0, scope=0)

        spn_text = "(0.6*((0.4*(Gaussian(V0|mean=0.0;stdev=1.0)) + 0.6*(Gaussian(V0|mean=2.0;stdev=1.0)))) + 0.4*(Gaussian(V0|mean=2.0;stdev=1.0)))"

        print(spn_to_str_equation(spn))

        self.assertEqual(spn_to_str_equation(spn), spn_text)

    def test_prod(self):
        spn = Gaussian(0.0, 1.0, scope=0) * Gaussian(2.0, 1.0, scope=1)

        spn_text = "(Gaussian(V0|mean=0.0;stdev=1.0) * Gaussian(V1|mean=2.0;stdev=1.0))"

        self.assertEqual(spn_to_str_equation(spn), spn_text)

    def test_spn(self):
        spn = 0.4 * (Gaussian(0.0, 1.0, scope=0) * Gaussian(2.0, 3.0, scope=1)) + \
              0.6 * (Gaussian(4.0, 5.0, scope=0) * Gaussian(6.0, 7.0, scope=1))

        spn_text = "(0.4*((Gaussian(V0|mean=0.0;stdev=1.0) * Gaussian(V1|mean=2.0;stdev=3.0))) + " + \
                   "0.6*((Gaussian(V0|mean=4.0;stdev=5.0) * Gaussian(V1|mean=6.0;stdev=7.0))))"

        self.assertEqual(spn_to_str_equation(spn), spn_text)


if __name__ == '__main__':
    unittest.main()
