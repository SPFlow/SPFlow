#!/usr/bin/env python3

import unittest
import torch
from spn.gpu.PyTorch.Param import Param


class TestParameterSpace(unittest.TestCase):
    @unittest.skip("This test should fail")
    def test_wrong_parameter_type(self):
        param = Param()
        assert param.add_param(10)

    def test_correct_parameter_type(self):
        param = Param()
        param.add_param(torch.nn.Parameter(torch.zeros(4, 2)))
        temp_tensor = torch.nn.Parameter(torch.zeros(4, 2))
        assert temp_tensor.shape == param.parameter_list[0].shape


if __name__ == "__main__":
    unittest.main()
