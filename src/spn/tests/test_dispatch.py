from spn.python.structure.nodes import Gaussian, ParametricLeaf
from spn.torch.structure.nodes import TorchGaussian

import unittest
import numpy as np
import torch

from multipledispatch import dispatch  # type: ignore


class TestDispatch(unittest.TestCase):
    def test_multiple_import(self):

        # create dummy nodes
        node = Gaussian([0], 0.0, 1.0)
        torch_node = TorchGaussian([0], 0.0, 1.0)

        # make sure that log likelihood is not known for either node
        # self.assertRaises(UnboundLocalError, log_likelihood(leaf_node))
        # self.assertRaises(UnboundLocalError, log_likelihood(torch_leaf_node))

        from spn.python.inference.nodes.node import log_likelihood

        # call log likelihood for python node
        log_likelihood(node, np.random.rand(1, 1))
        # verify that signature for torch node is not yet registered
        self.assertRaises(NotImplementedError, log_likelihood, torch_node, torch.rand(1, 1))

        # import log_likelihood from torch backend
        from spn.torch.inference import log_likelihood

        # make sure that both signatures are correctly registered
        log_likelihood(node, np.random.rand(1, 1))
        log_likelihood(torch_node, torch.rand(1, 1))

    def test_import_and_local_registartion(self):

        # define dummy node class
        class DummyNode(ParametricLeaf):
            pass

        # create dummy nodes
        node = Gaussian([0], 0.0, 1.0)
        dummy_node = DummyNode([0])

        from spn.python.inference.nodes.node import log_likelihood

        # call log likelihood for python node
        log_likelihood(node, np.random.rand(1, 1))
        # verify that signature for dummy node is not yet registered
        self.assertRaises(NotImplementedError, log_likelihood, dummy_node, np.random.rand(1, 1))

        @dispatch(DummyNode, np.ndarray)
        def log_likelihood(DummyNode: node, data: np.ndarray):
            pass

        # make sure that both signatures are correctly registered
        log_likelihood(node, np.random.rand(1, 1))
        log_likelihood(dummy_node, np.random.rand(1, 1))


if __name__ == "__main__":
    unittest.main()
