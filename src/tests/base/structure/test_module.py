from spflow.meta.data.scope import Scope
from spflow.base.structure.nodes.node import SPNSumNode
from .nodes.dummy_node import DummyNode
from .dummy_module import DummyModule, DummyNestedModule
import numpy as np
import unittest
import random


class TestNode(unittest.TestCase):
    def test_input_to_output_id_1(self):

        s = SPNSumNode(
            [
                DummyNode(Scope([0])),
                DummyNode(Scope([0])),
                DummyNode(Scope([0])),
                DummyNode(Scope([0])),
                DummyNode(Scope([0])),
            ]
        )

        self.assertEqual(s.input_to_output_ids([0]), ([0], [0]))
        self.assertEqual(s.input_to_output_ids([1]), ([1], [0]))
        self.assertEqual(s.input_to_output_ids([2]), ([2], [0]))
        self.assertEqual(s.input_to_output_ids([3]), ([3], [0]))
        self.assertEqual(s.input_to_output_ids([4]), ([4], [0]))

    def test_input_to_output_id_2(self):

        s = SPNSumNode(
            [
                DummyModule(n=2, scope=Scope([0])),
                DummyNode(Scope([0])),
                DummyModule(n=3, scope=Scope([0])),
            ]
        )

        self.assertEqual(s.input_to_output_ids([0]), ([0], [0]))
        self.assertEqual(s.input_to_output_ids([1]), ([0], [1]))
        self.assertEqual(s.input_to_output_ids([2]), ([1], [0]))
        self.assertEqual(s.input_to_output_ids([3]), ([2], [0]))
        self.assertEqual(s.input_to_output_ids([4]), ([2], [1]))
        self.assertEqual(s.input_to_output_ids([5]), ([2], [2]))

    def test_input_to_output_id_3(self):

        s = DummyNestedModule(
            children=[
                DummyModule(n=2, scope=Scope([0])),
                DummyNode(Scope([0])),
                DummyModule(n=3, scope=Scope([0])),
            ]
        )

        self.assertEqual(s.placeholders[0].input_to_output_ids([0]), ([0], [0]))
        self.assertEqual(s.placeholders[0].input_to_output_ids([1]), ([0], [1]))
        self.assertEqual(s.placeholders[0].input_to_output_ids([2]), ([1], [0]))
        self.assertEqual(s.placeholders[0].input_to_output_ids([3]), ([2], [0]))
        self.assertEqual(s.placeholders[0].input_to_output_ids([4]), ([2], [1]))
        self.assertEqual(s.placeholders[0].input_to_output_ids([5]), ([2], [2]))


if __name__ == "__main__":
    unittest.main()
