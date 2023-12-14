import unittest

from spflow.meta.data import Scope
from spflow.tensorly.structure.spn import SumNode
from spflow.tensorly.structure import Module
from spflow.tensorly.structure.general.nodes import Node
from .dummy_module import DummyModule, DummyNestedModule
from .general.node.dummy_node import DummyNode

tc = unittest.TestCase()

def test_input_to_output_id_1(do_for_all_backends):

    s = SumNode(
        [
            DummyNode(Scope([0])),
            DummyNode(Scope([0])),
            DummyNode(Scope([0])),
            DummyNode(Scope([0])),
            DummyNode(Scope([0])),
        ]
    )

    tc.assertEqual(s.input_to_output_ids([0]), ([0], [0]))
    tc.assertEqual(s.input_to_output_ids([1]), ([1], [0]))
    tc.assertEqual(s.input_to_output_ids([2]), ([2], [0]))
    tc.assertEqual(s.input_to_output_ids([3]), ([3], [0]))
    tc.assertEqual(s.input_to_output_ids([4]), ([4], [0]))

def test_input_to_output_id_2(do_for_all_backends):

    s = SumNode(
        [
            DummyModule(n=2, scope=Scope([0])),
            DummyNode(Scope([0])),
            DummyModule(n=3, scope=Scope([0])),
        ]
    )

    tc.assertEqual(s.input_to_output_ids([0]), ([0], [0]))
    tc.assertEqual(s.input_to_output_ids([1]), ([0], [1]))
    tc.assertEqual(s.input_to_output_ids([2]), ([1], [0]))
    tc.assertEqual(s.input_to_output_ids([3]), ([2], [0]))
    tc.assertEqual(s.input_to_output_ids([4]), ([2], [1]))
    tc.assertEqual(s.input_to_output_ids([5]), ([2], [2]))

def test_input_to_output_id_3(do_for_all_backends):

    s = DummyNestedModule(
        children=[
            DummyModule(n=2, scope=Scope([0])),
            DummyNode(Scope([0])),
            DummyModule(n=3, scope=Scope([0])),
        ]
    )

    tc.assertEqual(s.placeholders[0].input_to_output_ids([0]), ([0], [0]))
    tc.assertEqual(s.placeholders[0].input_to_output_ids([1]), ([0], [1]))
    tc.assertEqual(s.placeholders[0].input_to_output_ids([2]), ([1], [0]))
    tc.assertEqual(s.placeholders[0].input_to_output_ids([3]), ([2], [0]))
    tc.assertEqual(s.placeholders[0].input_to_output_ids([4]), ([2], [1]))
    tc.assertEqual(s.placeholders[0].input_to_output_ids([5]), ([2], [2]))


if __name__ == "__main__":
    unittest.main()
