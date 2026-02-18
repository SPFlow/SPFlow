import numpy as np
import pytest
import torch
from torch import nn

from spflow.meta import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.ops.split import Split, SplitMode
from spflow.modules.products.outer_product import OuterProduct


class _TinyLeaf(Module):
    def __init__(self, scope_idx: int, features: int = 2, channels: int = 1, reps: int = 1, ll_dims: int = 4):
        super().__init__()
        self.scope = Scope([scope_idx])
        self.in_shape = ModuleShape(features, channels, reps)
        self.out_shape = ModuleShape(features, channels, reps)
        self.ll_dims = ll_dims

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array(
            [Scope([self.scope.query[0]]) for _ in range(self.out_shape.features)], dtype=object
        ).reshape(self.out_shape.features, self.out_shape.repetitions)

    def log_likelihood(self, data, cache=None):
        b = data.shape[0]
        if self.ll_dims == 3:
            return torch.zeros((b, self.out_shape.features, self.out_shape.channels))
        if self.ll_dims == 5:
            return torch.zeros(
                (b, self.out_shape.features, self.out_shape.channels, self.out_shape.repetitions, 1)
            )
        return torch.zeros((b, self.out_shape.features, self.out_shape.channels, self.out_shape.repetitions))

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None):
        data = self._prepare_sample_data(num_samples, data)
        return torch.nan_to_num(data, nan=0.0)

    def _sample(self, data, sampling_ctx, cache):
        del sampling_ctx
        del cache
        return torch.nan_to_num(data, nan=0.0)

    def expectation_maximization(self, data, bias_correction=True, cache=None):
        return None

    def maximum_likelihood_estimation(
        self, data, weights=None, bias_correction=True, nan_strategy="ignore", cache=None
    ):
        return None

    def marginalize(self, marg_rvs, prune=True, cache=None):
        return self


class _UnknownSplit(Split):
    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([[Scope([0]), Scope([1])]], dtype=object)

    def merge_split_indices(self, *split_indices: torch.Tensor) -> torch.Tensor:
        return split_indices[0]

    def merge_split_tensors(self, *split_tensors: torch.Tensor) -> torch.Tensor:
        return split_tensors[0]

    def log_likelihood(self, data, cache=None):
        b = data.shape[0]
        return torch.zeros((b, 1, self.out_shape.channels, self.out_shape.repetitions))


def test_single_module_wrap_and_num_splits_validation():
    leaf = _TinyLeaf(0, features=4)
    node = OuterProduct(inputs=leaf, split_mode=SplitMode.consecutive(num_splits=2))
    assert node.input_is_split

    with pytest.raises(ValueError):
        OuterProduct(inputs=leaf, num_splits=1)


def test_split_num_splits_mismatch_raises():
    leaf = _TinyLeaf(0, features=4)
    split = _UnknownSplit(leaf, num_splits=2)
    node = OuterProduct(inputs=split, num_splits=2)
    node.num_splits = 3
    with pytest.raises(ValueError):
        node.check_shapes()


def test_feature_to_scope_split_branch_executes():
    leaf = _TinyLeaf(0, features=4)
    node = OuterProduct(inputs=leaf, num_splits=2)
    f2s = node.feature_to_scope
    assert f2s.shape[1] == node.out_shape.repetitions


def test_map_methods_raise_for_unknown_split_type():
    leaf = _TinyLeaf(0, features=4)
    split = _UnknownSplit(leaf, num_splits=2)
    node = OuterProduct(inputs=split, num_splits=2)

    with pytest.raises(NotImplementedError):
        node.map_out_channels_to_in_channels(torch.zeros((2, 1), dtype=torch.long))

    with pytest.raises(NotImplementedError):
        node.map_out_mask_to_in_mask(torch.ones((2, 1), dtype=torch.bool))


def test_log_likelihood_handles_3d_and_invalid_dims():
    a = _TinyLeaf(0, features=2, channels=2, ll_dims=3)
    b = _TinyLeaf(1, features=2, channels=2, ll_dims=3)
    node = OuterProduct(inputs=[a, b])
    out = node.log_likelihood(torch.zeros((2, 2)))
    assert out.ndim == 4

    bad_a = _TinyLeaf(0, features=2, channels=2, ll_dims=5)
    bad_b = _TinyLeaf(1, features=2, channels=2, ll_dims=5)
    bad = OuterProduct(inputs=[bad_a, bad_b])
    with pytest.raises(ValueError):
        bad.log_likelihood(torch.zeros((2, 2)))


def test_log_likelihood_repetitions_none_branch():
    a = _TinyLeaf(0, features=2, channels=2, ll_dims=3)
    b = _TinyLeaf(1, features=2, channels=2, ll_dims=3)
    node = OuterProduct(inputs=[a, b])
    node.out_shape = ModuleShape(node.out_shape.features, node.out_shape.channels, None)

    out = node.log_likelihood(torch.zeros((2, 2)))
    assert out.ndim == 3


def test_check_shapes_false_and_fast_true_paths(monkeypatch):
    node = OuterProduct(inputs=[_TinyLeaf(0), _TinyLeaf(1)])
    monkeypatch.setattr(node, "input_is_split", False)

    class _EmptyIterableInputs(nn.Module):
        def __init__(self, ref_module: Module):
            super().__init__()
            self.ref = ref_module

        def __getitem__(self, idx):
            return self.ref

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 1

    node.inputs = _EmptyIterableInputs(_TinyLeaf(0))
    assert node.check_shapes() is False

    node2 = OuterProduct(inputs=[_TinyLeaf(0, channels=2), _TinyLeaf(1, channels=2)])
    assert node2.check_shapes() is True

    node3 = OuterProduct(
        inputs=[_TinyLeaf(0, channels=1), _TinyLeaf(1, channels=2), _TinyLeaf(2, channels=1)]
    )
    assert node3.check_shapes() is True


def test_split_specific_scope_and_channel_mapping_paths():
    leaf = _TinyLeaf(0, features=4, channels=2)
    node_consec = OuterProduct(inputs=leaf, num_splits=2)

    # Force isinstance(self.inputs, Split) branch in feature_to_scope.
    split_module = node_consec.inputs[0]
    node_consec.inputs = split_module
    _ = node_consec.feature_to_scope
    node_consec.inputs = torch.nn.ModuleList([split_module])

    out_ids = torch.zeros((2, 2), dtype=torch.long)
    out_mask = torch.ones((2, 2), dtype=torch.bool)
    assert node_consec.map_out_channels_to_in_channels(out_ids).shape[-1] == 1
    assert node_consec.map_out_mask_to_in_mask(out_mask).shape[-1] == 1

    node_inter = OuterProduct(inputs=leaf, split_mode=SplitMode.interleaved(num_splits=2))
    assert node_inter.map_out_channels_to_in_channels(out_ids).shape[-1] == 1
    assert node_inter.map_out_mask_to_in_mask(out_mask).shape[-1] == 1
