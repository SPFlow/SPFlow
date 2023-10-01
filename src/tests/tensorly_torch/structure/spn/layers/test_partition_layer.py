import itertools
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.tensorly.structure import marginalize
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import Gaussian
from spflow.tensorly.structure.spn import PartitionLayer
from spflow.tensorly.structure.spn.layers.partition_layer import toLayerBased, toNodeBased
from spflow.tensorly.structure.spn.layers_layerbased.partition_layer import toLayerBased, toNodeBased, updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy, tl_cartesian_product

from ...general.nodes.dummy_node import DummyNode

tc = unittest.TestCase()

def test_partition_layer_initialization(do_for_all_backends):

    # dummy partitios over pair-wise disjont scopes
    input_partitions = [
        [DummyNode(Scope([0])), DummyNode(Scope([0]))],
        [
            DummyNode(Scope([1, 3])),
            DummyNode(Scope([1, 3])),
            DummyNode(Scope([1, 3])),
        ],
        [DummyNode(Scope([2]))],
    ]

    # ----- check attributes after correct initialization -----

    l = PartitionLayer(child_partitions=input_partitions)
    # make sure number of creates nodes is correct
    tc.assertEqual(l.n_out, np.prod([len(partition) for partition in input_partitions]))
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([0, 1, 2, 3]) for _ in range(l.n_out)]))
    # make sure order of nodes is correct (important)
    for indices, indices_torch in zip(
        itertools.product([0, 1], [2, 3, 4], [5]),
        tl_cartesian_product(tl.tensor([0, 1]), tl.tensor([2, 3, 4]), tl.tensor([5])),
    ):
        tc.assertTrue(tl.all(tl.tensor(indices) == indices_torch))

    # ----- no child partitions -----
    tc.assertRaises(ValueError, PartitionLayer, [])

    # ----- empty partition -----
    tc.assertRaises(ValueError, PartitionLayer, [[]])

    # ----- scopes inside partition differ -----
    tc.assertRaises(
        ValueError,
        PartitionLayer,
        [
            [DummyNode(Scope([0]))],
            [DummyNode(Scope([1])), DummyNode(Scope([2]))],
        ],
    )

    # ----- partitions of non-pair-wise disjoint scopes -----
    tc.assertRaises(
        ValueError,
        PartitionLayer,
        [
            [DummyNode(Scope([0]))],
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
        ],
    )

def test_partition_layer_structural_marginalization(do_for_all_backends):

    # dummy partitios over pair-wise disjont scopes
    input_partitions = [
        [DummyNode(Scope([0])), DummyNode(Scope([0]))],
        [
            DummyNode(Scope([1, 3])),
            DummyNode(Scope([1, 3])),
            DummyNode(Scope([1, 3])),
        ],
        [DummyNode(Scope([2]))],
    ]

    l = PartitionLayer(child_partitions=input_partitions)
    # should marginalize entire module
    l_marg = marginalize(l, [0, 1, 2, 3])
    tc.assertTrue(l_marg is None)
    # should marginalize entire partition
    l_marg = marginalize(l, [2])
    tc.assertTrue(l_marg.scope == Scope([0, 1, 3]))
    # should partially marginalize one partition
    l_marg = marginalize(l, [3])
    tc.assertTrue(l_marg.scope == Scope([0, 1, 2]))

    # ----- pruning -----
    l = PartitionLayer(child_partitions=input_partitions[1:])

    l_marg = marginalize(l, [1, 3], prune=True)
    tc.assertTrue(isinstance(l_marg, DummyNode))

def test_sum_layer_layerbased_conversion(do_for_all_backends):

    partition_layer = PartitionLayer(
        child_partitions=[
            [Gaussian(Scope([0])), Gaussian(Scope([0]))],
            [Gaussian(Scope([1]))],
            [
                Gaussian(Scope([2])),
                Gaussian(Scope([2])),
                Gaussian(Scope([2])),
            ],
        ]
    )

    layer_based_partition_layer = toLayerBased(partition_layer)
    tc.assertEqual(layer_based_partition_layer.n_out, partition_layer.n_out)
    node_based_partition_layer = toNodeBased(layer_based_partition_layer)
    tc.assertEqual(node_based_partition_layer.n_out, partition_layer.n_out)

    node_based_partition_layer2 = toNodeBased(partition_layer)
    tc.assertEqual(node_based_partition_layer2.n_out, partition_layer.n_out)
    layer_based_partition_layer2 = toLayerBased(layer_based_partition_layer)
    tc.assertEqual(layer_based_partition_layer2.n_out, partition_layer.n_out)

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    partition_layer = PartitionLayer(
        child_partitions=[
            [Gaussian(Scope([0])), Gaussian(Scope([0]))],
            [Gaussian(Scope([1]))],
            [
                Gaussian(Scope([2])),
                Gaussian(Scope([2])),
                Gaussian(Scope([2])),
            ],
        ]
    )
    n_out = partition_layer.n_out
    for backend in backends:
        with tl.backend_context(backend):
            partition_layer_updated = updateBackend(partition_layer)
            tc.assertTrue(n_out == partition_layer_updated.n_out)


if __name__ == "__main__":
    unittest.main()
