import numpy as np
import pytest
import torch

from spflow.exceptions import ShapeError
from spflow.meta import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.ops.split import Split, SplitMode
from spflow.modules.ops.split_consecutive import SplitConsecutive
from spflow.modules.products.elementwise_product import ElementwiseProduct


class _TinyLeaf(Module):
    def __init__(self, scope_idx: int, features: int = 2, channels: int = 1, reps: int = 1, return_4d: bool = True):
        super().__init__()
        self.scope = Scope([scope_idx])
        self.in_shape = ModuleShape(features, channels, reps)
        self.out_shape = ModuleShape(features, channels, reps)
        self.return_4d = return_4d

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([Scope([self.scope.query[0]]) for _ in range(self.out_shape.features)], dtype=object).reshape(
            self.out_shape.features, self.out_shape.repetitions
        )

    def log_likelihood(self, data, cache=None):
        b = data.shape[0]
        if self.return_4d:
            return torch.zeros((b, self.out_shape.features, self.out_shape.channels, self.out_shape.repetitions))
        return torch.zeros((b, self.out_shape.features, self.out_shape.channels))

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None):
        data = self._prepare_sample_data(num_samples, data)
        return torch.nan_to_num(data, nan=0.0)

    def expectation_maximization(self, data, bias_correction=True, cache=None):
        return None

    def maximum_likelihood_estimation(self, data, weights=None, bias_correction=True, nan_strategy="ignore", cache=None):
        return None

    def marginalize(self, marg_rvs, prune=True, cache=None):
        return self


class _UnknownSplit(Split):
    @property
    def feature_to_scope(self) -> np.ndarray:
        arr = np.array([Scope([0]), Scope([1])], dtype=object).reshape(1, 2)
        return arr

    def merge_split_indices(self, *split_indices: torch.Tensor) -> torch.Tensor:
        return split_indices[0]

    def log_likelihood(self, data, cache=None):
        b = data.shape[0]
        # (batch, out_features_per_split, channels, reps)
        return torch.zeros((b, 1, self.out_shape.channels, self.out_shape.repetitions))


def test_single_module_uses_split_mode_or_default_split():
    leaf = _TinyLeaf(0, features=4)

    with_mode = ElementwiseProduct(inputs=leaf, split_mode=SplitMode.interleaved(num_splits=2))
    default_mode = ElementwiseProduct(inputs=leaf, num_splits=2)

    assert with_mode.input_is_split
    assert default_mode.input_is_split


def test_check_shapes_rejects_split_num_splits_mismatch():
    leaf = _TinyLeaf(0, features=4)
    split = _UnknownSplit(leaf, num_splits=2)
    node = ElementwiseProduct(inputs=split, num_splits=2)
    node.num_splits = 3

    with pytest.raises(ValueError, match="num_splits must be the same"):
        node.check_shapes()


def test_check_shapes_returns_false_when_no_shapes(monkeypatch):
    a = _TinyLeaf(0)
    b = _TinyLeaf(1)
    node = ElementwiseProduct(inputs=[a, b])

    monkeypatch.setattr(node, "input_is_split", False)
    monkeypatch.setattr(node, "inputs", torch.nn.ModuleList())
    assert node.check_shapes() is False


def test_check_shapes_raises_for_unbroadcastable_shapes():
    node = ElementwiseProduct(inputs=[_TinyLeaf(0), _TinyLeaf(1)])
    a = _TinyLeaf(0, features=2, channels=2)
    b = _TinyLeaf(1, features=4, channels=3)
    node.inputs = torch.nn.ModuleList([a, b])
    with pytest.raises(ShapeError, match="not broadcastable"):
        node.check_shapes()


def test_map_methods_raise_for_unsupported_split_type():
    leaf = _TinyLeaf(0, features=4)
    split = _UnknownSplit(leaf, num_splits=2)
    node = ElementwiseProduct(inputs=split, num_splits=2)

    with pytest.raises(NotImplementedError):
        node.map_out_channels_to_in_channels(torch.zeros((2, 1), dtype=torch.long))

    with pytest.raises(NotImplementedError):
        node.map_out_mask_to_in_mask(torch.ones((2, 1), dtype=torch.bool))


def test_log_likelihood_expands_3d_channel_input():
    a = _TinyLeaf(0, features=2, channels=1, return_4d=False)
    b = _TinyLeaf(1, features=2, channels=2, return_4d=False)
    node = ElementwiseProduct(inputs=[a, b])

    out = node.log_likelihood(torch.zeros((3, 2)))
    assert out.shape == (3, node.out_shape.features, node.out_shape.channels, node.out_shape.repetitions)


def test_split_mode_branches_for_scope_and_channel_mapping():
    leaf = _TinyLeaf(0, features=4, channels=2)
    node_consec = ElementwiseProduct(inputs=leaf, num_splits=2)
    _ = node_consec.feature_to_scope

    out_ids = torch.zeros((2, 2), dtype=torch.long)
    out_mask = torch.ones((2, 2), dtype=torch.bool)
    assert node_consec.map_out_channels_to_in_channels(out_ids).shape[-1] == 1
    assert node_consec.map_out_mask_to_in_mask(out_mask).shape[-1] == 1

    node_inter = ElementwiseProduct(inputs=leaf, split_mode=SplitMode.interleaved(num_splits=2))
    assert node_inter.map_out_channels_to_in_channels(out_ids).shape[-1] == 1
    assert node_inter.map_out_mask_to_in_mask(out_mask).shape[-1] == 1


def test_out_f_equals_one_and_condition4_true_branch(monkeypatch):
    # out_f == 1 path
    leaf = _TinyLeaf(0, features=1, channels=1)
    node = ElementwiseProduct(inputs=leaf, num_splits=2)
    assert node.out_shape.features == 1

    # Artificial malformed shape tuples to reach condition-4 True branch.
    split = SplitConsecutive(_TinyLeaf(1, features=4), num_splits=2)
    node2 = ElementwiseProduct(inputs=split, num_splits=2)
    monkeypatch.setattr(node2.inputs[0], "get_out_shapes", lambda event_shape: [(1, 1), (1, 1, 1)])
    assert node2.check_shapes() is True
