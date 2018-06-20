import unittest

from spn.io.CPP import to_cpp
from spn.structure.Base import Leaf, bfs
from spn.structure.leaves.parametric.Parametric import Gaussian


class TestCPP(unittest.TestCase):

    def test_bcpp(self):
        D = Gaussian(mean=1.0, stdev=1.0, scope=[0])
        E = Gaussian(mean=2.0, stdev=2.0, scope=[1])
        F = Gaussian(mean=3.0, stdev=3.0, scope=[0])
        G = Gaussian(mean=4.0, stdev=4.0, scope=[1])

        B = D * E
        C = F * G

        A = 0.3 * B + 0.7 * C

        cpp_code = to_cpp(A)

        print(cpp_code)


if __name__ == '__main__':
    unittest.main()
