'''
Created on June 11, 2018

@author: Alejandro Molina
'''

import unittest

from spn.algorithms.TransformStructure import SPN_Reshape
from spn.structure.Base import Leaf, Product


class TestTransformation(unittest.TestCase):
    def test_sum(self):
        spn = Product()
        for s in range(7):
            spn.children.append(Leaf(scope=s))

        new_spn = SPN_Reshape(spn, 2)

        print(spn)


if __name__ == '__main__':
    unittest.main()
