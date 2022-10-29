from spflow.torch.structure.nodes.leaves.parametric.projections import (
    proj_bounded_to_real,
    proj_real_to_bounded,
)

import torch
import numpy as np

import random
import unittest


class TestTorchParametric(unittest.TestCase):
    def test_projections(self):

        eps = 0.0001

        # ----- bounded intervals -----

        lb = -1.0
        ub = 1.0

        x_real = torch.tensor([-10.0, 0.0, 10.0])
        x_bounded = torch.tensor([lb + eps, lb + (ub - lb) / 2.0, ub - eps])

        self.assertTrue(
            torch.allclose(
                proj_real_to_bounded(x_real, lb=lb, ub=ub), x_bounded
            )
        )
        self.assertTrue(
            torch.allclose(
                proj_bounded_to_real(x_bounded, lb=lb, ub=ub), x_real, rtol=0.1
            )
        )

        # ----- left bounded intervals -----

        x_real = torch.tensor([-10.0, 1.0])
        x_bounded = torch.tensor([lb + eps, 1.7])

        self.assertTrue(
            torch.allclose(
                proj_real_to_bounded(x_real, lb=lb), x_bounded, rtol=0.1
            )
        )
        self.assertTrue(
            torch.allclose(
                proj_bounded_to_real(x_bounded, lb=lb), x_real, rtol=0.1
            )
        )

        # ----- right bounded intervals -----

        x_real = torch.tensor([-10.0, 1.0])
        x_bounded = torch.tensor([ub - eps, -1.7])

        self.assertTrue(
            torch.allclose(
                proj_real_to_bounded(x_real, ub=ub), x_bounded, rtol=0.1
            )
        )
        self.assertTrue(
            torch.allclose(
                proj_bounded_to_real(x_bounded, ub=ub), x_real, rtol=0.1
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
