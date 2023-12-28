import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.node.leaf.uniform import Uniform as UniformBase
from spflow.base.structure.general.layer.leaf.uniform import UniformLayer as UniformLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import marginalize
from spflow.torch.structure.general.node.leaf.uniform import Uniform as UniformTorch
from spflow.torch.structure.general.layer.leaf.uniform import UniformLayer as UniformLayerTorch
from spflow.torch.structure.general.layer.leaf.uniform import updateBackend

from spflow.structure import AutoLeaf
from spflow.modules.layer import UniformLayer
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_layer_initialization(do_for_all_backends):
    # ----- check attributes after correct initialization -----
    start_values = [0.5, 0.3, 1.0]
    end_values = [1.3, 1.0, 1.2]
    support_outside_values = [True, False, True]
    l = UniformLayer(
        scope=Scope([1]),
        n_nodes=3,
        start=start_values,
        end=end_values,
        support_outside=support_outside_values,
    )
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
    # make sure parameter properties works correctly
    for (
        start_layer_node,
        end_layer_node,
        support_outside_layer_node,
        start_value,
        end_value,
        support_outside_value,
    ) in zip(
        l.start,
        l.end,
        l.support_outside,
        start_values,
        end_values,
        support_outside_values,
    ):
        tc.assertTrue(np.allclose(tle.toNumpy(start_layer_node), start_value))
        tc.assertTrue(np.allclose(tle.toNumpy(end_layer_node), end_value))

    # ----- float/int parameter values -----
    start_value = 0.73
    end_value = 1.9
    support_outside_value = False
    l = UniformLayer(
        scope=Scope([1]),
        n_nodes=3,
        start=start_value,
        end=end_value,
        support_outside=support_outside_value,
    )

    for start_layer_node, end_layer_node, support_outside_layer_node in zip(
        l.start, l.end, l.support_outside
    ):
        tc.assertTrue(np.allclose(tle.toNumpy(start_layer_node), start_value))
        tc.assertTrue(np.allclose(tle.toNumpy(end_layer_node), end_value))
        tc.assertTrue(
            np.allclose(
                support_outside_layer_node,
                tl.tensor(support_outside_value),
            )
        )

    # ----- list parameter values -----
    start_values = [0.17, 0.8, 0.53]
    end_values = [0.9, 1.34, 0.98]
    support_outside_values = [True, False, False]
    l = UniformLayer(
        scope=Scope([1]),
        n_nodes=3,
        start=start_values,
        end=end_values,
        support_outside=support_outside_values,
    )

    for (
        start_layer_node,
        end_layer_node,
        support_outside_layer_node,
        start_value,
        end_value,
        support_outside_value,
    ) in zip(
        l.start,
        l.end,
        l.support_outside,
        start_values,
        end_values,
        support_outside_values,
    ):
        tc.assertTrue(np.allclose(tle.toNumpy(start_layer_node), start_value))
        tc.assertTrue(np.allclose(tle.toNumpy(end_layer_node), end_value))
        tc.assertTrue(
            np.allclose(
                support_outside_layer_node,
                tl.tensor(support_outside_value),
            )
        )

    # wrong number of values
    tc.assertRaises(
        ValueError,
        UniformLayer,
        Scope([0]),
        start_values[:-1],
        end_values,
        support_outside_values,
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        UniformLayer,
        Scope([0]),
        start_values,
        end_values[:-1],
        support_outside_values[:-1],
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        UniformLayer,
        Scope([0]),
        start_values,
        end_values,
        support_outside_values[:-1],
        n_nodes=3,
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        UniformLayer,
        Scope([0]),
        [start_values for _ in range(3)],
        end_values,
        support_outside_values,
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        UniformLayer,
        Scope([0]),
        start_values,
        [end_values for _ in range(3)],
        support_outside_values,
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        UniformLayer,
        Scope([0]),
        start_values,
        end_values,
        [support_outside_values for _ in range(3)],
        n_nodes=3,
    )

    # ----- numpy parameter values -----

    l = UniformLayer(
        scope=Scope([1]),
        n_nodes=3,
        start=np.array(start_values),
        end=np.array(end_values),
        support_outside=np.array(support_outside_values),
    )

    for (
        start_layer_node,
        end_layer_node,
        support_outside_layer_node,
        start_value,
        end_value,
        support_outside_value,
    ) in zip(
        l.start,
        l.end,
        l.support_outside,
        start_values,
        end_values,
        support_outside_values,
    ):
        tc.assertTrue(np.allclose(tle.toNumpy(start_layer_node), start_value))
        tc.assertTrue(np.allclose(tle.toNumpy(end_layer_node), end_value))
        tc.assertTrue(
            np.allclose(
                support_outside_layer_node,
                tl.tensor(support_outside_value),
            )
        )

    # wrong number of values
    tc.assertRaises(
        ValueError,
        UniformLayer,
        Scope([0]),
        np.array(start_values[:-1]),
        np.array(end_values),
        np.array(support_outside_values),
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        UniformLayer,
        Scope([0]),
        np.array(start_values),
        np.array(end_values[:-1]),
        np.array(support_outside_values),
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        UniformLayer,
        Scope([0]),
        np.array(start_values),
        np.array(end_values),
        np.array(support_outside_values[:-1]),
        n_nodes=3,
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        UniformLayer,
        Scope([0]),
        np.array([start_values for _ in range(3)]),
        np.array(end_values),
        np.array(support_outside_values),
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        UniformLayer,
        Scope([0]),
        np.array(start_values),
        np.array([end_values for _ in range(3)]),
        np.array(support_outside_values),
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        UniformLayer,
        Scope([0]),
        np.array(start_values),
        np.array(end_values),
        np.array([support_outside_values for _ in range(3)]),
        n_nodes=3,
    )

    # ---- different scopes -----
    l = UniformLayer(scope=Scope([1]), n_nodes=3, start=0.0, end=1.0)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, UniformLayer, Scope([0]), n_nodes=0, start=0.0, end=1.0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, UniformLayer, Scope([]), n_nodes=3, start=0.0, end=1.0)
    tc.assertRaises(ValueError, UniformLayer, [], n_nodes=3, start=0.0, end=1.0)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1]), Scope([0]), Scope([0])]
    l = UniformLayer(scope=[Scope([1]), Scope([0])], n_nodes=3, start=0.0, end=1.0)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)


def test_accept(do_for_all_backends):
    # discrete meta type (should reject)
    tc.assertFalse(
        UniformLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # Uniform feature type instance
    tc.assertTrue(
        UniformLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Uniform(start=-1.0, end=2.0)]),
                FeatureContext(Scope([1]), [FeatureTypes.Uniform(start=1.0, end=3.0)]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        UniformLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Uniform(-1.0, 2.0)]),
            ]
        )
    )

    # conditional scope
    tc.assertFalse(
        UniformLayer.accepts(
            [
                FeatureContext(
                    Scope([0], [1]),
                    [FeatureTypes.Uniform(start=-1.0, end=2.0)],
                )
            ]
        )
    )

    # multivariate signature
    tc.assertFalse(
        UniformLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [
                        FeatureTypes.Uniform(start=-1.0, end=2.0),
                        FeatureTypes.Uniform(start=-1.0, end=2.0),
                    ],
                )
            ]
        )
    )


def test_initialization_from_signatures(do_for_all_backends):
    uniform = UniformLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Uniform(start=-1.0, end=2.0)]),
            FeatureContext(Scope([1]), [FeatureTypes.Uniform(start=1.0, end=3.0)]),
        ]
    )
    tc.assertTrue(np.allclose(uniform.start, tl.tensor([-1.0, 1.0])))
    tc.assertTrue(np.allclose(uniform.end, tl.tensor([2.0, 3.0])))
    tc.assertTrue(uniform.scopes_out == [Scope([0]), Scope([1])])

    # ----- invalid arguments -----

    # discrete meta type
    tc.assertRaises(
        ValueError,
        UniformLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # invalid feature type
    tc.assertRaises(
        ValueError,
        UniformLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        UniformLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        UniformLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )


def test_autoleaf(do_for_all_backends):
    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(UniformLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        UniformLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Uniform(start=-1.0, end=2.0)]),
                FeatureContext(Scope([1]), [FeatureTypes.Uniform(start=1.0, end=3.0)]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    uniform = AutoLeaf(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Uniform(start=-1.0, end=2.0)]),
            FeatureContext(Scope([1]), [FeatureTypes.Uniform(start=1.0, end=3.0)]),
        ]
    )
    tc.assertTrue(np.allclose(uniform.start, tl.tensor([-1.0, 1.0])))
    tc.assertTrue(np.allclose(uniform.end, tl.tensor([2.0, 3.0])))
    tc.assertTrue(uniform.scopes_out == [Scope([0]), Scope([1])])


def test_layer_structural_marginalization(do_for_all_backends):
    if tl.get_backend() == "numpy":
        UniformInst = UniformBase
        UniformInstLayer = UniformLayerBase
    elif tl.get_backend() == "pytorch":
        UniformInst = UniformTorch
        UniformInstLayer = UniformLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # ---------- same scopes -----------

    l = UniformLayer(
        scope=Scope([1]),
        start=[0.73, 0.29],
        end=[1.41, 0.9],
        support_outside=[True, False],
        n_nodes=2,
    )

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
    tc.assertTrue(np.allclose(l.start, l_marg.start))
    tc.assertTrue(np.allclose(l.end, l_marg.end))
    tc.assertTrue(np.allclose(l.support_outside, l_marg.support_outside))

    # ---------- different scopes -----------

    l = UniformLayer(
        scope=[Scope([1]), Scope([0])],
        start=[0.73, 0.29],
        end=[1.41, 0.9],
        support_outside=[True, False],
    )

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, UniformInst))
    tc.assertEqual(l_marg.scope, Scope([0]))
    tc.assertTrue(np.allclose(l_marg.start, tl.tensor(0.29)))
    tc.assertTrue(np.allclose(l_marg.end, tl.tensor(0.9)))
    tc.assertTrue(tl.all(l_marg.support_outside == tl.tensor(False)))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, UniformInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)
    tc.assertTrue(np.allclose(l_marg.start, tl.tensor(0.29)))
    tc.assertTrue(np.allclose(l_marg.end, tl.tensor(0.9)))
    tc.assertTrue(tl.all(l_marg.support_outside == tl.tensor(False)))

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
    tc.assertTrue(np.allclose(l.start, l_marg.start))
    tc.assertTrue(np.allclose(l.end, l_marg.end))
    tc.assertTrue(np.allclose(l.support_outside, l_marg.support_outside))


def test_layer_dist(do_for_all_backends):
    start_values = [0.73, 0.29, 0.5]
    end_values = [0.9, 1.34, 0.98]
    support_outside_values = [True, False, True]
    l = UniformLayer(
        scope=Scope([1]),
        start=start_values,
        end=end_values,
        support_outside=support_outside_values,
        n_nodes=3,
    )

    # ----- full dist -----
    dist = l.dist()

    if tl.get_backend() == "numpy":
        start_list = [d.kwds.get("loc") for d in dist]
        end_list = []
        for i in range(len(dist)):
            end_list.append(dist[i].kwds.get("scale") + start_list[i])
    elif tl.get_backend() == "pytorch":
        start_list = dist.low
        end_list = dist.high
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for start_value, end_value, start_dist, end_dist in zip(start_values, end_values, start_list, end_list):
        tc.assertTrue(np.allclose(tl.tensor(start_value), start_dist))
        tc.assertTrue(np.allclose(tl.tensor(end_value), end_dist))

    # ----- partial dist -----
    dist = l.dist([1, 2])

    if tl.get_backend() == "numpy":
        start_list = [d.kwds.get("loc") for d in dist]
        end_list = []
        for i in range(len(dist)):
            end_list.append(dist[i].kwds.get("scale") + start_list[i])
    elif tl.get_backend() == "pytorch":
        start_list = dist.low
        end_list = dist.high
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for start_value, end_value, start_dist, end_dist in zip(
        start_values[1:], end_values[1:], start_list, end_list
    ):
        tc.assertTrue(np.allclose(tl.tensor(start_value), start_dist))
        tc.assertTrue(np.allclose(tl.tensor(end_value), end_dist))

    dist = l.dist([1, 0])

    if tl.get_backend() == "numpy":
        start_list = [d.kwds.get("loc") for d in dist]
        end_list = []
        for i in range(len(dist)):
            end_list.append(dist[i].kwds.get("scale") + start_list[i])
    elif tl.get_backend() == "pytorch":
        start_list = dist.low
        end_list = dist.high
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for start_value, end_value, start_dist, end_dist in zip(
        reversed(start_values[:-1]),
        reversed(end_values[:-1]),
        start_list,
        end_list,
    ):
        tc.assertTrue(np.allclose(tl.tensor(start_value), start_dist))
        tc.assertTrue(np.allclose(tl.tensor(end_value), end_dist))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    uniform = UniformLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        start=[0.2, -0.9, 0.31],
        end=[0.3, 1.0, 0.5],
        support_outside=[True, False, True],
    )
    for backend in backends:
        with tl.backend_context(backend):
            uniform_updated = updateBackend(uniform)
            tc.assertTrue(np.all(uniform.scopes_out == uniform_updated.scopes_out))
            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*uniform.get_params()[0]]),
                    np.array([*uniform_updated.get_params()[0]]),
                )
            )

            tc.assertTrue(
                np.allclose(
                    np.array([*uniform.get_params()[1]]),
                    np.array([*uniform_updated.get_params()[1]]),
                )
            )

            tc.assertTrue(
                np.allclose(
                    np.array([*uniform.get_params()[2]]),
                    np.array([*uniform_updated.get_params()[2]]),
                )
            )


def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    uniform_default = UniformLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        start=[0.2, -0.9, 0.31],
        end=[0.3, 1.0, 0.5],
        support_outside=[True, False, True],
    )
    tc.assertTrue(uniform_default.dtype == tl.float32)
    tc.assertTrue(uniform_default.start.dtype == tl.float32)
    tc.assertTrue(uniform_default.end.dtype == tl.float32)
    # tc.assertTrue(uniform_default.support_outside.dtype == tl.float32)

    # change to float64 model
    uniform_updated = UniformLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        start=[0.2, -0.9, 0.31],
        end=[0.3, 1.0, 0.5],
        support_outside=[True, False, True],
    )
    uniform_updated.to_dtype(tl.float64)
    tc.assertTrue(uniform_updated.dtype == tl.float64)
    tc.assertTrue(uniform_updated.start.dtype == tl.float64)
    tc.assertTrue(uniform_updated.end.dtype == tl.float64)
    # tc.assertTrue(uniform_updated.support_outside.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array([*uniform_default.get_params()]),
            np.array([*uniform_updated.get_params()]),
        )
    )


def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    uniform_default = UniformLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        start=[0.2, -0.9, 0.31],
        end=[0.3, 1.0, 0.5],
        support_outside=[True, False, True],
    )
    uniform_updated = UniformLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        start=[0.2, -0.9, 0.31],
        end=[0.3, 1.0, 0.5],
        support_outside=[True, False, True],
    )
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, uniform_updated.to_device, cuda)
        return

    # put model on gpu
    uniform_updated.to_device(cuda)

    tc.assertTrue(uniform_default.device.type == "cpu")
    tc.assertTrue(uniform_updated.device.type == "cuda")

    tc.assertTrue(uniform_default.start.device.type == "cpu")
    tc.assertTrue(uniform_updated.start.device.type == "cuda")
    tc.assertTrue(uniform_default.end.device.type == "cpu")
    tc.assertTrue(uniform_updated.end.device.type == "cuda")
    tc.assertTrue(uniform_default.support_outside.device.type == "cpu")
    tc.assertTrue(uniform_updated.support_outside.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array([*uniform_default.get_params()]),
            np.array([*uniform_updated.get_params()]),
        )
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
