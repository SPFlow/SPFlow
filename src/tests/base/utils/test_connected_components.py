from spflow.base.utils import connected_components

import numpy as np
import unittest
import random


class TestNode(unittest.TestCase):
    def test_cc_initialization(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # not symmetric adjacency matrix
        self.assertRaises(ValueError, connected_components, np.tri(3))
        # symmetric adjacency matrix
        adj_mat = np.random.randint(0, 2, (3, 3))
        connected_components((adj_mat + adj_mat.T) / 2)

    def test_cc_single_component(self):

        adj_mat = np.array(
            [
                [0, 1, 1, 0, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        )

        ccs = connected_components(adj_mat + adj_mat.T)

        self.assertTrue(len(ccs) == 1)
        self.assertTrue(ccs[0] == {0, 1, 2, 3, 4})

    def test_cc_multiple_components(self):

        adj_mat = np.array(
            [
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        )

        ccs = connected_components(adj_mat + adj_mat.T)

        self.assertTrue(len(ccs) == 3)
        self.assertTrue(ccs[0] == {0, 3, 4})
        self.assertTrue(ccs[1] == {1})
        self.assertTrue(ccs[2] == {2})


if __name__ == "__main__":
    unittest.main()
