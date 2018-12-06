"""
Created on December 06, 2018

@author: Alejandro Molina
"""
import unittest

from spn.algorithms.Validity import is_valid

from spn.algorithms.TransformStructure import Compress
from spn.structure.leaves.parametric.Parametric import Gaussian


class TestCompression(unittest.TestCase):
    def test_compression_leaves(self):
        C1 = Gaussian(mean=1, stdev=0, scope=0)
        C2 = Gaussian(mean=1, stdev=0, scope=0)

        A = 0.7 * C1 + 0.3 * C2

        Compress(A)

        self.assertTrue(*is_valid(A))
        self.assertEqual(id(A.children[0]), id(A.children[1]))

        C1 = Gaussian(mean=1, stdev=0, scope=0)
        C2 = Gaussian(mean=1, stdev=0, scope=1)

        B = C1 * C2

        Compress(B)
        self.assertTrue(*is_valid(B))

        self.assertNotEqual(id(B.children[0]), id(B.children[1]))

    def test_compression_leaves_deeper(self):
        C1 = Gaussian(mean=1, stdev=0, scope=0)
        C2 = Gaussian(mean=1, stdev=1, scope=1)
        C3 = Gaussian(mean=1, stdev=0, scope=0)
        C4 = Gaussian(mean=2, stdev=0, scope=1)

        R = 0.4 * (C1 * C2) + 0.6 * (C3 * C4)

        Compress(R)
        self.assertTrue(*is_valid(R))

        self.assertNotEqual(id(R.children[0]), id(R.children[1]))
        self.assertEqual(id(R.children[0].children[0]), id(C1))
        self.assertEqual(id(R.children[0].children[1]), id(C2))
        self.assertEqual(id(R.children[1].children[0]), id(C1))
        self.assertEqual(id(R.children[1].children[1]), id(C4))

    def test_compression_internal_nodes(self):
        C1 = Gaussian(mean=1, stdev=0, scope=0)
        C2 = Gaussian(mean=1, stdev=1, scope=1)
        C3 = Gaussian(mean=1, stdev=0, scope=0)
        C4 = Gaussian(mean=1, stdev=1, scope=1)

        R = 0.4 * (C1 * C2) + 0.6 * (C3 * C4)

        Compress(R)
        self.assertTrue(*is_valid(R))

        self.assertEqual(id(R.children[0]), id(R.children[1]))
        self.assertEqual(id(R.children[0].children[0]), id(C1))
        self.assertEqual(id(R.children[0].children[1]), id(C2))


if __name__ == "__main__":
    unittest.main()
