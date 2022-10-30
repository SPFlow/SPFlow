from spflow.torch.structure.layers.leaves.parametric.uniform import (
    UniformLayer,
    marginalize,
    toTorch,
    toBase,
)
from spflow.torch.structure.nodes.leaves.parametric.uniform import Uniform
from spflow.base.structure.layers.leaves.parametric.uniform import (
    UniformLayer as BaseUniformLayer,
)
from spflow.meta.data.scope import Scope
import torch
import numpy as np
import unittest
import itertools


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_initialization(self):

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
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])])
        )
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
            self.assertTrue(
                torch.allclose(start_layer_node, torch.tensor(start_value))
            )
            self.assertTrue(
                torch.allclose(end_layer_node, torch.tensor(end_value))
            )

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
            self.assertTrue(
                torch.allclose(start_layer_node, torch.tensor(start_value))
            )
            self.assertTrue(
                torch.allclose(end_layer_node, torch.tensor(end_value))
            )
            self.assertTrue(
                torch.allclose(
                    support_outside_layer_node,
                    torch.tensor(support_outside_value),
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
            self.assertTrue(
                torch.allclose(start_layer_node, torch.tensor(start_value))
            )
            self.assertTrue(
                torch.allclose(end_layer_node, torch.tensor(end_value))
            )
            self.assertTrue(
                torch.allclose(
                    support_outside_layer_node,
                    torch.tensor(support_outside_value),
                )
            )

        # wrong number of values
        self.assertRaises(
            ValueError,
            UniformLayer,
            Scope([0]),
            start_values[:-1],
            end_values,
            support_outside_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            UniformLayer,
            Scope([0]),
            start_values,
            end_values[:-1],
            support_outside_values[:-1],
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            UniformLayer,
            Scope([0]),
            start_values,
            end_values,
            support_outside_values[:-1],
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            UniformLayer,
            Scope([0]),
            [start_values for _ in range(3)],
            end_values,
            support_outside_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            UniformLayer,
            Scope([0]),
            start_values,
            [end_values for _ in range(3)],
            support_outside_values,
            n_nodes=3,
        )
        self.assertRaises(
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
            self.assertTrue(
                torch.allclose(start_layer_node, torch.tensor(start_value))
            )
            self.assertTrue(
                torch.allclose(end_layer_node, torch.tensor(end_value))
            )
            self.assertTrue(
                torch.allclose(
                    support_outside_layer_node,
                    torch.tensor(support_outside_value),
                )
            )

        # wrong number of values
        self.assertRaises(
            ValueError,
            UniformLayer,
            Scope([0]),
            np.array(start_values[:-1]),
            np.array(end_values),
            np.array(support_outside_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            UniformLayer,
            Scope([0]),
            np.array(start_values),
            np.array(end_values[:-1]),
            np.array(support_outside_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            UniformLayer,
            Scope([0]),
            np.array(start_values),
            np.array(end_values),
            np.array(support_outside_values[:-1]),
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            UniformLayer,
            Scope([0]),
            np.array([start_values for _ in range(3)]),
            np.array(end_values),
            np.array(support_outside_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            UniformLayer,
            Scope([0]),
            np.array(start_values),
            np.array([end_values for _ in range(3)]),
            np.array(support_outside_values),
            n_nodes=3,
        )
        self.assertRaises(
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
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(
            ValueError, UniformLayer, Scope([0]), n_nodes=0, start=0.0, end=1.0
        )

        # ----- invalid scope -----
        self.assertRaises(
            ValueError, UniformLayer, Scope([]), n_nodes=3, start=0.0, end=1.0
        )
        self.assertRaises(
            ValueError, UniformLayer, [], n_nodes=3, start=0.0, end=1.0
        )

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = UniformLayer(
            scope=[Scope([1]), Scope([0])], n_nodes=3, start=0.0, end=1.0
        )

        for layer_scope, node_scope in zip(l.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = UniformLayer(
            scope=Scope([1]),
            start=[0.73, 0.29],
            end=[1.41, 0.9],
            support_outside=[True, False],
            n_nodes=2,
        )

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(torch.allclose(l.start, l_marg.start))
        self.assertTrue(torch.allclose(l.end, l_marg.end))
        self.assertTrue(
            torch.allclose(l.support_outside, l_marg.support_outside)
        )

        # ---------- different scopes -----------

        l = UniformLayer(
            scope=[Scope([1]), Scope([0])],
            start=[0.73, 0.29],
            end=[1.41, 0.9],
            support_outside=[True, False],
        )

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Uniform))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertTrue(torch.allclose(l_marg.start, torch.tensor(0.29)))
        self.assertTrue(torch.allclose(l_marg.end, torch.tensor(0.9)))
        self.assertTrue(
            torch.all(l_marg.support_outside == torch.tensor(False))
        )

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, UniformLayer))
        self.assertEqual(len(l_marg.scopes_out), 1)
        self.assertTrue(torch.allclose(l_marg.start, torch.tensor(0.29)))
        self.assertTrue(torch.allclose(l_marg.end, torch.tensor(0.9)))
        self.assertTrue(
            torch.all(l_marg.support_outside == torch.tensor(False))
        )

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(torch.allclose(l.start, l_marg.start))
        self.assertTrue(torch.allclose(l.end, l_marg.end))
        self.assertTrue(
            torch.allclose(l.support_outside, l_marg.support_outside)
        )

    def test_layer_dist(self):

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

        for start_value, end_value, start_dist, end_dist in zip(
            start_values, end_values, dist.low, dist.high
        ):
            self.assertTrue(
                torch.allclose(torch.tensor(start_value), start_dist)
            )
            self.assertTrue(torch.allclose(torch.tensor(end_value), end_dist))

        # ----- partial dist -----
        dist = l.dist([1, 2])

        for start_value, end_value, start_dist, end_dist in zip(
            start_values[1:], end_values[1:], dist.low, dist.high
        ):
            self.assertTrue(
                torch.allclose(torch.tensor(start_value), start_dist)
            )
            self.assertTrue(torch.allclose(torch.tensor(end_value), end_dist))

        dist = l.dist([1, 0])

        for start_value, end_value, start_dist, end_dist in zip(
            reversed(start_values[:-1]),
            reversed(end_values[:-1]),
            dist.low,
            dist.high,
        ):
            self.assertTrue(
                torch.allclose(torch.tensor(start_value), start_dist)
            )
            self.assertTrue(torch.allclose(torch.tensor(end_value), end_dist))

    def test_layer_backend_conversion_1(self):

        torch_layer = UniformLayer(
            scope=[Scope([0]), Scope([1]), Scope([0])],
            start=[0.2, -0.9, 0.31],
            end=[0.3, 1.0, 0.5],
            support_outside=[True, False, True],
        )
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(
            np.allclose(base_layer.start, torch_layer.start.numpy())
        )
        self.assertTrue(np.allclose(base_layer.end, torch_layer.end.numpy()))
        self.assertTrue(
            np.allclose(
                base_layer.support_outside, torch_layer.support_outside.numpy()
            )
        )
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

    def test_layer_backend_conversion_2(self):

        base_layer = BaseUniformLayer(
            scope=[Scope([0]), Scope([1]), Scope([0])],
            start=[0.2, -0.9, 0.31],
            end=[0.3, 1.0, 0.5],
            support_outside=[True, False, True],
        )
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(
            np.allclose(base_layer.start, torch_layer.start.numpy())
        )
        self.assertTrue(np.allclose(base_layer.end, torch_layer.end.numpy()))
        self.assertTrue(
            np.allclose(
                base_layer.support_outside, torch_layer.support_outside.numpy()
            )
        )
        self.assertEqual(base_layer.n_out, torch_layer.n_out)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
