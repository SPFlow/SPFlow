import numpy as np
import pytest
import torch

from spflow.exceptions import InvalidParameterError, ShapeError, UnsupportedOperationError
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.normal import Normal
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.sums.signed_sum import SignedSum
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext


class _SignedInput(Module):
    def __init__(self, bad_signed_shape: bool = False):
        super().__init__()
        self.scope = Scope([0])
        self.in_shape = ModuleShape(features=1, channels=1, repetitions=1)
        self.out_shape = ModuleShape(features=1, channels=2, repetitions=1)
        self.bad_signed_shape = bad_signed_shape
        self.last_sampling_ctx = None

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([[Scope([0])]], dtype=object)

    def log_likelihood(self, data: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        return torch.zeros((data.shape[0], 1, 2, 1), dtype=torch.get_default_dtype())

    def signed_logabs_and_sign(
        self, data: torch.Tensor, cache: Cache | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bad_signed_shape:
            return (
                torch.zeros((data.shape[0], 1, 2), dtype=torch.get_default_dtype()),
                torch.ones((data.shape[0], 1, 2), dtype=torch.int8),
            )
        return (
            torch.zeros((data.shape[0], 1, 2, 1), dtype=torch.get_default_dtype()),
            torch.ones((data.shape[0], 1, 2, 1), dtype=torch.int8),
        )

    def sample(
        self,
        num_samples: int | None = None,
        data: torch.Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx=None,
    ) -> torch.Tensor:
        self.last_sampling_ctx = sampling_ctx
        data[:, self.scope.query] = 0.0
        return data

    def _sample(
        self,
        data: torch.Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
    ) -> torch.Tensor:
        del cache
        self.last_sampling_ctx = sampling_ctx
        data[:, self.scope.query] = 0.0
        return data

    def marginalize(
        self, marg_rvs: list[int], prune: bool = True, cache: Cache | None = None
    ) -> Module | None:
        return self


def _sampling_ctx(batch_size: int, num_features: int = 1) -> SamplingContext:
    return SamplingContext(
        channel_index=torch.zeros((batch_size, num_features), dtype=torch.long),
        mask=torch.ones((batch_size, num_features), dtype=torch.bool),
    )


def test_signed_sum_signed_eval_shapes_and_finiteness():
    leaf_a = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    leaf_b = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)

    # Cat(dim=2) will create in_channels=2
    w = torch.tensor([[[[1.0]], [[-0.5]]]])  # (F=1, IC=2, OC=1, R=1)
    node = SignedSum(inputs=[leaf_a, leaf_b], out_channels=1, num_repetitions=1, weights=w)

    x = torch.randn(7, 1)
    cache = Cache()
    logabs, sign = node.signed_logabs_and_sign(x, cache=cache)

    assert logabs.shape == (7, 1, 1, 1)
    assert sign.shape == (7, 1, 1, 1)
    assert torch.isfinite(logabs).all()
    assert ((sign == -1) | (sign == 0) | (sign == 1)).all()


def test_signed_sum_init_validates_inputs_and_weights():
    leaf = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)

    with pytest.raises(ValueError):
        SignedSum(inputs=[])

    with pytest.raises(ShapeError):
        SignedSum(inputs=leaf, out_channels=1, num_repetitions=1, weights=torch.ones((1, 2, 1, 1)))


def test_signed_sum_init_single_input_and_repr_and_scope_mapping():
    leaf = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    node = SignedSum(inputs=[leaf], out_channels=1, num_repetitions=1, weights=None)

    assert node.inputs is leaf
    assert node.feature_to_scope.shape == leaf.feature_to_scope.shape
    assert f"weights={node.weights_shape}" in node.extra_repr()


def test_signed_sum_init_validates_inputs_and_weight_shape():
    leaf = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    with pytest.raises(ValueError):
        SignedSum(inputs=[])
    with pytest.raises(ShapeError):
        SignedSum(inputs=leaf, out_channels=1, num_repetitions=1, weights=torch.ones((1, 2, 1, 1)))


def test_signed_sum_init_single_item_input_list_and_repr():
    leaf = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    node = SignedSum(inputs=[leaf], out_channels=1, num_repetitions=1, weights=None)

    assert node.inputs is leaf
    assert tuple(node.weights.shape) == node.weights_shape
    assert node.feature_to_scope.shape == leaf.feature_to_scope.shape
    assert f"weights={node.weights_shape}" in node.extra_repr()


def test_signed_sum_unsupported_operations_raise():
    leaf = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    node = SignedSum(inputs=leaf, out_channels=1, num_repetitions=1, weights=torch.ones((1, 1, 1, 1)))
    x = torch.randn(3, 1)

    with pytest.raises(UnsupportedOperationError):
        node.log_likelihood(x)
    with pytest.raises(UnsupportedOperationError):
        node._expectation_maximization_step(x, cache=Cache())
    with pytest.raises(AttributeError):
        node.maximum_likelihood_estimation(x)
    with pytest.raises(UnsupportedOperationError):
        node.marginalize([0])


def test_signed_sum_unsupported_methods_raise():
    leaf = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    node = SignedSum(inputs=leaf, out_channels=1, num_repetitions=1, weights=torch.ones((1, 1, 1, 1)))
    x = torch.randn(2, 1)

    with pytest.raises(UnsupportedOperationError):
        node.log_likelihood(x)
    with pytest.raises(UnsupportedOperationError):
        node._expectation_maximization_step(x, cache=Cache())
    with pytest.raises(AttributeError):
        node.maximum_likelihood_estimation(x)
    with pytest.raises(UnsupportedOperationError):
        node.marginalize([0])


def test_signed_sum_signed_eval_uses_cache_and_signed_child_path():
    child = _SignedInput()
    node = SignedSum(
        inputs=child, out_channels=1, num_repetitions=1, weights=torch.tensor([[[[0.3]], [[-0.7]]]])
    )
    x = torch.randn(4, 1)
    cache = Cache()

    out0 = node.signed_logabs_and_sign(x, cache=cache)
    out1 = node.signed_logabs_and_sign(x, cache=cache)

    assert out0[0].shape == (4, 1, 1, 1)
    assert out0[1].shape == (4, 1, 1, 1)
    assert out0[0].data_ptr() == out1[0].data_ptr()
    assert out0[1].data_ptr() == out1[1].data_ptr()


def test_signed_sum_signed_eval_cache_hit_uses_cached_result():
    child = _SignedInput()
    node = SignedSum(
        inputs=child, out_channels=1, num_repetitions=1, weights=torch.tensor([[[[0.4]], [[0.6]]]])
    )
    x = torch.randn(3, 1)
    cache = Cache()

    first = node.signed_logabs_and_sign(x, cache=cache)
    second = node.signed_logabs_and_sign(x, cache=cache)
    assert first[0].data_ptr() == second[0].data_ptr()
    assert first[1].data_ptr() == second[1].data_ptr()


def test_signed_sum_signed_eval_raises_for_invalid_child_shape():
    child = _SignedInput(bad_signed_shape=True)
    node = SignedSum(
        inputs=child, out_channels=1, num_repetitions=1, weights=torch.tensor([[[[0.2]], [[0.8]]]])
    )

    with pytest.raises(ShapeError):
        node.signed_logabs_and_sign(torch.randn(2, 1))


def test_signed_sum_signed_eval_without_cache_and_bad_child_shape_raises():
    child = _SignedInput(bad_signed_shape=True)
    node = SignedSum(
        inputs=child, out_channels=1, num_repetitions=1, weights=torch.tensor([[[[0.2]], [[0.8]]]])
    )

    with pytest.raises(ShapeError):
        node.signed_logabs_and_sign(torch.randn(2, 1), cache=Cache())


@pytest.mark.parametrize("is_mpe", [True, False])
def test_signed_sum_sample_success_paths(is_mpe: bool):
    child = _SignedInput()
    node = SignedSum(
        inputs=child, out_channels=1, num_repetitions=1, weights=torch.tensor([[[[0.9]], [[0.1]]]])
    )

    samples = node.sample(num_samples=5)

    assert samples.shape == (5, 1)
    assert torch.isfinite(samples).all()
    assert child.last_sampling_ctx is not None
    assert child.last_sampling_ctx.channel_index.shape == (5, 1)
    assert child.last_sampling_ctx.mask.shape == (5, 1)


def test_signed_sum_sample_rejects_evidence():
    child = _SignedInput()
    node = SignedSum(
        inputs=child, out_channels=1, num_repetitions=1, weights=torch.tensor([[[[0.9]], [[0.1]]]])
    )
    evidence = torch.tensor([[0.0], [float("nan")]])

    with pytest.raises(UnsupportedOperationError):
        node.sample(data=evidence)


def test_signed_sum_sample_rejects_evidence_and_negative_weights():
    child = _SignedInput()
    node_pos = SignedSum(
        inputs=child, out_channels=1, num_repetitions=1, weights=torch.tensor([[[[0.9]], [[0.1]]]])
    )
    with pytest.raises(UnsupportedOperationError):
        node_pos.sample(data=torch.tensor([[0.0], [float("nan")]]))

    node_neg = SignedSum(
        inputs=child, out_channels=1, num_repetitions=1, weights=torch.tensor([[[[1.0]], [[-0.1]]]])
    )
    with pytest.raises(UnsupportedOperationError):
        node_neg.sample(num_samples=3)


def test_signed_sum_sample_rejects_negative_weights():
    child = _SignedInput()
    node = SignedSum(
        inputs=child, out_channels=1, num_repetitions=1, weights=torch.tensor([[[[1.0]], [[-0.1]]]])
    )

    with pytest.raises(UnsupportedOperationError):
        node.sample(num_samples=3)


def test_signed_sum_sample_rejects_non_unit_repetitions():
    child = _SignedInput()
    weights = torch.ones((1, 2, 1, 2), dtype=torch.get_default_dtype())
    node = SignedSum(inputs=child, out_channels=1, num_repetitions=2, weights=weights)

    with pytest.raises(UnsupportedOperationError):
        node.sample(num_samples=3)


def test_signed_sum_sample_rejects_mismatched_mask_width_internal_context():
    child = _SignedInput()
    node = SignedSum(
        inputs=child, out_channels=1, num_repetitions=1, weights=torch.tensor([[[[0.9]], [[0.1]]]])
    )
    sampling_ctx = SamplingContext(
        channel_index=torch.zeros((2, 1), dtype=torch.long),
        mask=torch.ones((2, 1), dtype=torch.bool),
        repetition_index=torch.zeros((2,), dtype=torch.long),
    )
    sampling_ctx._mask = torch.ones((2, 2), dtype=torch.bool)  # type: ignore[attr-defined]
    with pytest.raises(InvalidParameterError, match="mismatched channel_index/mask shapes"):
        node._sample(
            data=torch.full((2, 1), float("nan")),
            sampling_ctx=sampling_ctx,
            cache=Cache(),
        )


def test_signed_sum_sample_rejects_bad_weight_dim_and_repetitions():
    child = _SignedInput()
    rep_node = SignedSum(inputs=child, out_channels=1, num_repetitions=2, weights=torch.ones((1, 2, 1, 2)))
    with pytest.raises(UnsupportedOperationError):
        rep_node.sample(num_samples=2)

    bad_dim_node = SignedSum(
        inputs=child, out_channels=1, num_repetitions=1, weights=torch.tensor([[[[0.7]], [[0.3]]]])
    )
    bad_dim_node.weights = torch.nn.Parameter(bad_dim_node.weights[..., 0])
    with pytest.raises(ShapeError):
        bad_dim_node.sample(num_samples=2)


def test_signed_sum_sample_rejects_non_4d_weights():
    child = _SignedInput()
    node = SignedSum(
        inputs=child, out_channels=1, num_repetitions=1, weights=torch.tensor([[[[0.7]], [[0.3]]]])
    )
    node.weights = torch.nn.Parameter(node.weights[..., 0])

    with pytest.raises(ShapeError):
        node.sample(num_samples=3)


def test_signed_sum_sample_defaults_to_single_sample():
    child = _SignedInput()
    node = SignedSum(
        inputs=child, out_channels=1, num_repetitions=1, weights=torch.tensor([[[[0.6]], [[0.4]]]])
    )

    samples = node.sample()

    assert samples.shape == (1, 1)
    assert torch.isfinite(samples).all()
