import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.utils import connected_components

tc = unittest.TestCase()

def test_cc_initialization(do_for_all_backends):

    # set seed
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # not symmetric adjacency matrix
    tc.assertRaises(ValueError, connected_components, tl.tensor(np.tri(3)))
    # symmetric adjacency matrix
    adj_mat = tl.tensor(np.random.randint(0, 2, (3, 3)))
    connected_components((adj_mat + adj_mat.T) / 2)

def test_cc_single_component(do_for_all_backends):

    adj_mat = tl.tensor(
        [
            [0, 1, 1, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )

    ccs = connected_components(adj_mat + adj_mat.T)

    tc.assertTrue(len(ccs) == 1)
    tc.assertTrue(ccs[0] == {0, 1, 2, 3, 4})

def test_cc_multiple_components(do_for_all_backends):

    adj_mat = tl.tensor(
        [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )

    ccs = connected_components(adj_mat + adj_mat.T)

    tc.assertTrue(len(ccs) == 3)
    tc.assertTrue(ccs[0] == {0, 3, 4})
    tc.assertTrue(ccs[1] == {1})
    tc.assertTrue(ccs[2] == {2})


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
