import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.node.leaf.hypergeometric import Hypergeometric as HypergeometricBase
from spflow.base.structure.general.layer.leaf.hypergeometric import HypergeometricLayer as HypergeometricLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import marginalize
from spflow.torch.structure.general.node.leaf.hypergeometric import Hypergeometric as HypergeometricTorch
from spflow.torch.structure.general.layer.leaf.hypergeometric import HypergeometricLayer as HypergeometricLayerTorch
from spflow.torch.structure.general.layer.leaf.hypergeometric import updateBackend

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layer.leaf.general_hypergeometric import HypergeometricLayer
from spflow.tensorly.structure.general.node.leaf.general_hypergeometric import Hypergeometric
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_initialization(do_for_all_backends):

    # ----- check attributes after correct initialization -----
    N_values = [10, 5, 7]
    M_values = [8, 2, 6]
    n_values = [3, 4, 5]
    l = HypergeometricLayer(
        scope=[Scope([1]), Scope([0]), Scope([2])],
        N=N_values,
        M=M_values,
        n=n_values,
    )
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([0]), Scope([2])]))
    # make sure parameter properties works correctly
    for (
        N_layer_node,
        M_layer_node,
        n_layer_node,
        N_value,
        M_value,
        n_value,
    ) in zip(l.N, l.M, l.n, N_values, M_values, n_values):
        tc.assertTrue(np.allclose(tl_toNumpy(N_layer_node), N_value))
        tc.assertTrue(np.allclose(tl_toNumpy(M_layer_node), M_value))
        tc.assertTrue(np.allclose(tl_toNumpy(n_layer_node), n_value))

    # ----- float/int parameter values -----
    N_value = 6
    M_value = 4
    n_value = 5
    l = HypergeometricLayer(scope=Scope([1]), n_nodes=3, N=N_value, M=M_value, n=n_value)

    for N_layer_node, M_layer_node, n_layer_node in zip(l.N, l.M, l.n):
        tc.assertTrue(np.allclose(tl_toNumpy(N_layer_node), N_value))
        tc.assertTrue(np.allclose(tl_toNumpy(M_layer_node), M_value))
        tc.assertTrue(np.allclose(tl_toNumpy(n_layer_node), n_value))

    # ----- list parameter values -----
    N_values = [10, 5, 7]
    M_values = [8, 2, 6]
    n_values = [3, 4, 5]
    l = HypergeometricLayer(
        scope=[Scope([0]), Scope([1]), Scope([2])],
        N=N_values,
        M=M_values,
        n=n_values,
    )

    for (
        N_layer_node,
        M_layer_node,
        n_layer_node,
        N_value,
        M_value,
        n_value,
    ) in zip(l.N, l.M, l.n, N_values, M_values, n_values):
        tc.assertTrue(np.allclose(tl_toNumpy(N_layer_node), N_value))
        tc.assertTrue(np.allclose(tl_toNumpy(M_layer_node), M_value))
        tc.assertTrue(np.allclose(tl_toNumpy(n_layer_node), n_value))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        HypergeometricLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        N_values[:-1],
        M_values,
        n_values,
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        HypergeometricLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        N_values,
        M_values[:-1],
        n_values,
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        HypergeometricLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        N_values,
        M_values,
        n_values[:-1],
        n_nodes=3,
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        HypergeometricLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        [N_values for _ in range(3)],
        M_values,
        n_values,
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        HypergeometricLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        N_values,
        [M_values for _ in range(3)],
        n_values,
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        HypergeometricLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        N_values,
        M_values,
        [n_values for _ in range(3)],
        n_nodes=3,
    )

    # ----- numpy parameter values -----

    l = HypergeometricLayer(
        scope=[Scope([0]), Scope([1]), Scope([2])],
        N=np.array(N_values),
        M=np.array(M_values),
        n=np.array(n_values),
    )

    for (
        N_layer_node,
        M_layer_node,
        n_layer_node,
        N_value,
        M_value,
        n_value,
    ) in zip(l.N, l.M, l.n, N_values, M_values, n_values):
        tc.assertTrue(np.allclose(tl_toNumpy(N_layer_node), N_value))
        tc.assertTrue(np.allclose(tl_toNumpy(M_layer_node), M_value))
        tc.assertTrue(np.allclose(tl_toNumpy(n_layer_node), n_value))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        HypergeometricLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        np.array(N_values[:-1]),
        np.array(M_values),
        np.array(n_values),
    )
    tc.assertRaises(
        ValueError,
        HypergeometricLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        np.array(N_values),
        np.array(M_values[:-1]),
        np.array(n_values),
    )
    tc.assertRaises(
        ValueError,
        HypergeometricLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        np.array(N_values),
        np.array(M_values),
        np.array(n_values[:-1]),
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        HypergeometricLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        np.array([N_values for _ in range(3)]),
        np.array(M_values),
        np.array(n_values),
    )
    tc.assertRaises(
        ValueError,
        HypergeometricLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        np.array(N_values),
        np.array([M_values for _ in range(3)]),
        np.array(n_values),
    )
    tc.assertRaises(
        ValueError,
        HypergeometricLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        np.array(N_values),
        np.array(M_values),
        np.array([n_values for _ in range(3)]),
    )

    # ---- different scopes -----
    l = HypergeometricLayer(scope=Scope([1]), n_nodes=3, N=5, M=3, n=2)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(
        ValueError,
        HypergeometricLayer,
        Scope([0]),
        n_nodes=0,
        N=5,
        M=3,
        n=2,
    )

    # ----- invalid scope -----
    tc.assertRaises(ValueError, HypergeometricLayer, Scope([]), n_nodes=3, N=5, M=3, n=2)
    tc.assertRaises(ValueError, HypergeometricLayer, [], n_nodes=3, N=5, M=3, n=2)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1]), Scope([0]), Scope([0])]
    l = HypergeometricLayer(scope=[Scope([1]), Scope([0])], n_nodes=3, N=5, M=3, n=2)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

def test_accept(do_for_all_backends):

    # discrete meta type (should reject)
    tc.assertFalse(
        HypergeometricLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # feature type instance
    tc.assertTrue(
        HypergeometricLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)]),
                FeatureContext(Scope([1]), [FeatureTypes.Hypergeometric(N=6, M=5, n=4)]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        HypergeometricLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1]), [FeatureTypes.Hypergeometric(N=6, M=5, n=4)]),
            ]
        )
    )

    # conditional scope
    tc.assertFalse(
        HypergeometricLayer.accepts(
            [
                FeatureContext(
                    Scope([0], [1]),
                    [FeatureTypes.Hypergeometric(N=4, M=2, n=3)],
                )
            ]
        )
    )

    # multivariate signature
    tc.assertFalse(
        HypergeometricLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [
                        FeatureTypes.Hypergeometric(N=4, M=2, n=3),
                        FeatureTypes.Hypergeometric(N=4, M=2, n=3),
                    ],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    hypergeometric = HypergeometricLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)]),
            FeatureContext(Scope([1]), [FeatureTypes.Hypergeometric(N=6, M=5, n=4)]),
        ]
    )
    tc.assertTrue(tl.all(hypergeometric.N == tl.tensor([4, 6])))
    tc.assertTrue(tl.all(hypergeometric.M == tl.tensor([2, 5])))
    tc.assertTrue(tl.all(hypergeometric.n == tl.tensor([3, 4])))
    tc.assertTrue(hypergeometric.scopes_out == [Scope([0]), Scope([1])])

    # ----- invalid arguments -----

    # discrete meta type
    tc.assertRaises(
        ValueError,
        HypergeometricLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # invalid feature type
    tc.assertRaises(
        ValueError,
        HypergeometricLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        HypergeometricLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        Hypergeometric.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        HypergeometricInstLayer = HypergeometricLayerBase
    elif tl.get_backend() == "pytorch":
        HypergeometricInstLayer = HypergeometricLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(HypergeometricLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        HypergeometricLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)]),
                FeatureContext(Scope([1]), [FeatureTypes.Hypergeometric(N=6, M=5, n=4)]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    hypergeometric = AutoLeaf(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)]),
            FeatureContext(Scope([1]), [FeatureTypes.Hypergeometric(N=6, M=5, n=4)]),
        ]
    )
    tc.assertTrue(isinstance(hypergeometric, HypergeometricInstLayer))
    tc.assertTrue(tl.all(hypergeometric.N == tl.tensor([4, 6])))
    tc.assertTrue(tl.all(hypergeometric.M == tl.tensor([2, 5])))
    tc.assertTrue(tl.all(hypergeometric.n == tl.tensor([3, 4])))
    tc.assertTrue(hypergeometric.scopes_out == [Scope([0]), Scope([1])])

def test_layer_structural_marginalization(do_for_all_backends):

    if tl.get_backend() == "numpy":
        HypergeometricInst = HypergeometricBase
        HypergeometricInstLayer = HypergeometricLayerBase
    elif tl.get_backend() == "pytorch":
        HypergeometricInst = HypergeometricTorch
        HypergeometricInstLayer = HypergeometricLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # ---------- same scopes -----------

    l = HypergeometricLayer(scope=Scope([1]), n_nodes=2, N=5, M=3, n=4)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.N), tl_toNumpy(l_marg.N)))
    tc.assertTrue(np.allclose(tl_toNumpy(l.M), tl_toNumpy(l_marg.M)))
    tc.assertTrue(np.allclose(tl_toNumpy(l.n), tl_toNumpy(l_marg.n)))

    # ---------- different scopes -----------

    l = HypergeometricLayer(scope=[Scope([1]), Scope([0])], N=[5, 7], M=[3, 6], n=[4, 3])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, HypergeometricInst))
    tc.assertEqual(l_marg.scope, Scope([0]))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.N), tl.tensor(7)))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.M), tl.tensor(6)))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.n), tl.tensor(3)))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, HypergeometricInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.N), tl.tensor(7)))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.M), tl.tensor(6)))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.n), tl.tensor(3)))

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.N), tl_toNumpy(l_marg.N)))
    tc.assertTrue(np.allclose(tl_toNumpy(l.M), tl_toNumpy(l_marg.M)))
    tc.assertTrue(np.allclose(tl_toNumpy(l.n), tl_toNumpy(l_marg.n)))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    hypergeometric = HypergeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
        N=[10, 5, 10],
        M=[8, 2, 8],
        n=[3, 4, 3])
    for backend in backends:
        with tl.backend_context(backend):
            hypergeometric_updated = updateBackend(hypergeometric)
            tc.assertTrue(np.all(hypergeometric.scopes_out == hypergeometric_updated.scopes_out))
            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*hypergeometric.get_params()[0]]),
                    np.array([*hypergeometric_updated.get_params()[0]]),
                )
            )

            tc.assertTrue(
                np.allclose(
                    np.array([*hypergeometric.get_params()[1]]),
                    np.array([*hypergeometric_updated.get_params()[1]]),
                )
            )

            tc.assertTrue(
                np.allclose(
                    np.array([*hypergeometric.get_params()[2]]),
                    np.array([*hypergeometric_updated.get_params()[2]]),
                )
            )
"""
def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    hypergeometric_default = HypergeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
                                         N=[10, 5, 10],
                                         M=[8, 2, 8],
                                         n=[3, 4, 3])
    tc.assertTrue(hypergeometric_default.dtype == tl.float32)
    tc.assertTrue(hypergeometric_default.N.dtype == tl.float32)
    tc.assertTrue(hypergeometric_default.M.dtype == tl.float32)
    tc.assertTrue(hypergeometric_default.n.dtype == tl.float32)

    # change to float64 model
    hypergeometric_updated = HypergeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
                                         N=[10, 5, 10],
                                         M=[8, 2, 8],
                                         n=[3, 4, 3])
    hypergeometric_updated.to_dtype(tl.float64)
    tc.assertTrue(hypergeometric_updated.dtype == tl.float64)
    tc.assertTrue(hypergeometric_updated.N.dtype == tl.float64)
    tc.assertTrue(hypergeometric_updated.M.dtype == tl.float64)
    tc.assertTrue(hypergeometric_updated.n.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array([*hypergeometric_default.get_params()]),
            np.array([*hypergeometric_updated.get_params()]),
        )
    )
"""

def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    hypergeometric_default = HypergeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
                                         N=[10, 5, 10],
                                         M=[8, 2, 8],
                                         n=[3, 4, 3])
    hypergeometric_updated = HypergeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
                                         N=[10, 5, 10],
                                         M=[8, 2, 8],
                                         n=[3, 4, 3])
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, hypergeometric_updated.to_device, cuda)
        return

    # put model on gpu
    hypergeometric_updated.to_device(cuda)

    tc.assertTrue(hypergeometric_default.device.type == "cpu")
    tc.assertTrue(hypergeometric_updated.device.type == "cuda")

    tc.assertTrue(hypergeometric_default.N.device.type == "cpu")
    tc.assertTrue(hypergeometric_updated.N.device.type == "cuda")
    tc.assertTrue(hypergeometric_default.M.device.type == "cpu")
    tc.assertTrue(hypergeometric_updated.M.device.type == "cuda")
    tc.assertTrue(hypergeometric_default.n.device.type == "cpu")
    tc.assertTrue(hypergeometric_updated.n.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array([*hypergeometric_default.get_params()]),
            np.array([*hypergeometric_updated.get_params()]),
        )
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
