import unittest

import numpy as np
import tensorly as tl
import torch

from spflow.base.structure.spn import Gaussian as BaseGaussian
from spflow.base.structure.spn import HadamardLayer as BaseHadamardLayer
from spflow.meta.data import Scope
#from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.tensorly.structure import marginalize
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import Gaussian
from spflow.tensorly.structure.spn import HadamardLayer
from spflow.tensorly.structure.spn.layers.hadamard_layer import toLayerBased, toNodeBased
from spflow.tensorly.structure.spn.layers_layerbased.hadamard_layer import toLayerBased, toNodeBased, updateBackend

from ...general.nodes.dummy_node import DummyNode


class TestNode(unittest.TestCase):
    def test_hadamard_layer_initialization(self):

        # dummy partitios over pair-wise disjont scopes
        input_partitions = [
            [DummyNode(Scope([0]))],
            [
                DummyNode(Scope([1, 3])),
                DummyNode(Scope([1, 3])),
                DummyNode(Scope([1, 3])),
            ],
            [DummyNode(Scope([2]))],
            [
                DummyNode(Scope([4])),
                DummyNode(Scope([4])),
                DummyNode(Scope([4])),
            ],
        ]

        # ----- check attributes after correct initialization -----

        l = HadamardLayer(child_partitions=input_partitions)
        # make sure number of creates nodes is correct
        self.assertEqual(l.n_out, 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([0, 1, 2, 3, 4]) for _ in range(l.n_out)]))

        # only one partition
        l = HadamardLayer(
            child_partitions=[
                [
                    DummyNode(Scope([1, 3])),
                    DummyNode(Scope([1, 3])),
                    DummyNode(Scope([1, 3])),
                ]
            ]
        )
        # make sure number of creates nodes is correct
        self.assertEqual(l.n_out, 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1, 3]) for _ in range(l.n_out)]))

        # ----- no child partitions -----
        self.assertRaises(ValueError, HadamardLayer, [])

        # ----- empty partition -----
        self.assertRaises(ValueError, HadamardLayer, [[]])

        # ----- scopes inside partition differ -----
        self.assertRaises(
            ValueError,
            HadamardLayer,
            [
                [DummyNode(Scope([0]))],
                [DummyNode(Scope([1])), DummyNode(Scope([2]))],
            ],
        )

        # ----- partitions of non-pair-wise disjoint scopes -----
        self.assertRaises(
            ValueError,
            HadamardLayer,
            [
                [DummyNode(Scope([0]))],
                [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            ],
        )

        # ----- invalid total outputs of partitions -----
        self.assertRaises(
            ValueError,
            HadamardLayer,
            [
                [DummyNode(Scope([0])), DummyNode(Scope([0]))],
                [
                    DummyNode(Scope([1])),
                    DummyNode(Scope([1])),
                    DummyNode(Scope([1])),
                ],
            ],
        )

    def test_hadamard_layer_structural_marginalization(self):

        # dummy partitios over pair-wise disjont scopes
        input_partitions = [
            [DummyNode(Scope([0]))],
            [
                DummyNode(Scope([1, 3])),
                DummyNode(Scope([1, 3])),
                DummyNode(Scope([1, 3])),
            ],
            [DummyNode(Scope([2]))],
            [
                DummyNode(Scope([4])),
                DummyNode(Scope([4])),
                DummyNode(Scope([4])),
            ],
        ]

        l = HadamardLayer(child_partitions=input_partitions)
        # should marginalize entire module
        l_marg = marginalize(l, [0, 1, 2, 3, 4])
        self.assertTrue(l_marg is None)
        # should marginalize entire partition
        l_marg = marginalize(l, [2])
        self.assertTrue(l_marg.scope == Scope([0, 1, 3, 4]))
        # should partially marginalize one partition
        l_marg = marginalize(l, [3])
        self.assertTrue(l_marg.scope == Scope([0, 1, 2, 4]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [5])
        self.assertTrue(
            l_marg.scopes_out
            == [
                Scope([0, 1, 2, 3, 4]),
                Scope([0, 1, 2, 3, 4]),
                Scope([0, 1, 2, 3, 4]),
            ]
        )

        # ----- pruning -----
        l = HadamardLayer(child_partitions=input_partitions[:2])

        l_marg = marginalize(l, [1, 3], prune=True)
        self.assertTrue(isinstance(l_marg, DummyNode))
    """
    def test_hadamard_layer_backend_conversion_1(self):

        torch_hadamard_layer = HadamardLayer(
            child_partitions=[
                [Gaussian(Scope([0])), Gaussian(Scope([0]))],
                [Gaussian(Scope([1]))],
                [Gaussian(Scope([2])), Gaussian(Scope([2]))],
            ]
        )

        base_hadamard_layer = toBase(torch_hadamard_layer)
        self.assertEqual(base_hadamard_layer.n_out, torch_hadamard_layer.n_out)

    def test_hadamard_layer_backend_conversion_2(self):

        base_hadamard_layer = BaseHadamardLayer(
            child_partitions=[
                [BaseGaussian(Scope([0])), BaseGaussian(Scope([0]))],
                [BaseGaussian(Scope([1]))],
                [BaseGaussian(Scope([2])), BaseGaussian(Scope([2]))],
            ]
        )

        torch_hadamard_layer = toTorch(base_hadamard_layer)
        self.assertEqual(base_hadamard_layer.n_out, torch_hadamard_layer.n_out)
    """
    def test_sum_layer_layerbased_conversion(self):

        hadamard_layer = HadamardLayer(
            child_partitions=[
                [Gaussian(Scope([0])), Gaussian(Scope([0]))],
                [Gaussian(Scope([1]))],
                [Gaussian(Scope([2])), Gaussian(Scope([2]))],
            ]
        )

        layer_based_hadamard_layer = toLayerBased(hadamard_layer)
        self.assertEqual(layer_based_hadamard_layer.n_out, hadamard_layer.n_out)
        node_based_hadamard_layer = toNodeBased(layer_based_hadamard_layer)
        self.assertEqual(node_based_hadamard_layer.n_out, hadamard_layer.n_out)

        node_based_hadamard_layer2 = toNodeBased(hadamard_layer)
        self.assertEqual(node_based_hadamard_layer2.n_out, hadamard_layer.n_out)
        layer_based_hadamard_layer2 = toLayerBased(layer_based_hadamard_layer)
        self.assertEqual(layer_based_hadamard_layer2.n_out, hadamard_layer.n_out)

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        hadamard_layer = HadamardLayer(
            child_partitions=[
                [Gaussian(Scope([0])), Gaussian(Scope([0]))],
                [Gaussian(Scope([1]))],
                [Gaussian(Scope([2])), Gaussian(Scope([2]))],
            ]
        )
        n_out = hadamard_layer.n_out
        for backend in backends:
            tl.set_backend(backend)
            hadamard_layer_updated = updateBackend(hadamard_layer)
            self.assertTrue(n_out == hadamard_layer_updated.n_out)


if __name__ == "__main__":
    unittest.main()
