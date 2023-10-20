import unittest

import numpy as np

from spflow.base.inference import likelihood, log_likelihood
from spflow.meta.data import Scope

from ..structure.dummy_module import DummyNestedModule
from ..structure.general.nodes.dummy_node import DummyNode


class TestModule(unittest.TestCase):
    def test_likelihood(self):

        dummy_nodes = [DummyNode(Scope([0]))]
        dummy_module = DummyNestedModule(children=dummy_nodes)

        dummy_data = np.array([[np.nan, 0.0, 1.0]])

        self.assertRaises(
            LookupError,
            log_likelihood,
            dummy_module.placeholders[0],
            dummy_data,
        )
        self.assertRaises(LookupError, likelihood, dummy_module.placeholders[0], dummy_data)


if __name__ == "__main__":
    unittest.main()