import numpy as np
import pytest
import torch
from torch import nn

from spflow.exceptions import ShapeError, UnsupportedOperationError
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.normal import Normal
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.utils.cache import Cache
from spflow.zoo.sos import SignedSum


class DummyInput(Module):
    def __init__(
        self,
        out_features: int = 1,
        out_channels: int = 2,
        num_repetitions: int = 1,
        signed_eval_dim: int = 4,
    ) -> None:
        super().__init__()
        self.scope = Scope(list(range(out_features)))
        self.in_shape = ModuleShape(features=out_features, channels=1, repetitions=1)
        self.out_shape = ModuleShape(
            features=out_features, channels=out_channels, repetitions=num_repetitions
        )
        self._feature_to_scope = np.array(
            [[Scope([i])] * num_repetitions for i in range(out_features)],
            dtype=object,
        )
        self.signed_calls = 0
        self.sample_calls = 0
        self.signed_eval_dim = signed_eval_dim

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self._feature_to_scope

    def log_likelihood(self, data: torch.Tensor, cache=None) -> torch.Tensor:
        batch_size = data.shape[0]
        return torch.zeros(
            batch_size,
            self.out_shape.features,
            self.out_shape.channels,
            self.out_shape.repetitions,
            dtype=data.dtype,
            device=data.device,
        )

    def signed_logabs_and_sign(self, data: torch.Tensor, cache=None) -> tuple[torch.Tensor, torch.Tensor]:
        self.signed_calls += 1
        batch_size = data.shape[0]
        if self.signed_eval_dim == 4:
            logabs = torch.zeros(
                batch_size,
                self.out_shape.features,
                self.out_shape.channels,
                self.out_shape.repetitions,
                dtype=data.dtype,
                device=data.device,
            )
            sign = torch.ones_like(logabs, dtype=torch.int8)
            return logabs, sign

        # Broken shape for error-path coverage.
        logabs_bad = torch.zeros(
            batch_size,
            self.out_shape.features,
            self.out_shape.channels,
            dtype=data.dtype,
            device=data.device,
        )
        sign_bad = torch.ones_like(logabs_bad, dtype=torch.int8)
        return logabs_bad, sign_bad

    def sample(
        self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None
    ) -> torch.Tensor:
        self.sample_calls += 1
        if data is None:
            data = torch.full((1, len(self.scope.query)), torch.nan)
        data[:, self.scope.query] = 0.0
        return data

    def expectation_maximization(self, data, bias_correction=True, cache=None) -> None:
        return None

    def maximum_likelihood_estimation(self, data, weights=None, cache=None) -> None:
        return None

    def marginalize(self, marg_rvs, prune=True, cache=None):
        return self


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


def test_signed_sum_init_single_input_and_repr_and_scope_mapping():
    leaf = Normal(scope=Scope([0]), out_channels=2, num_repetitions=1)
    node = SignedSum(inputs=[leaf], out_channels=1, num_repetitions=1, weights=None)

    assert node.inputs is leaf
    assert np.array_equal(node.feature_to_scope, leaf.feature_to_scope)
    assert "weights=(1, 2, 1, 1)" in node.extra_repr()


def test_signed_sum_init_validates_inputs_and_weight_shape():
    leaf = Normal(scope=Scope([0]), out_channels=2, num_repetitions=1)

    with pytest.raises(ValueError):
        SignedSum(inputs=[])

    with pytest.raises(ShapeError):
        SignedSum(inputs=leaf, out_channels=1, num_repetitions=1, weights=torch.ones(1, 3, 1, 1))


def test_signed_sum_unsupported_methods_raise():
    leaf = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    node = SignedSum(inputs=leaf, out_channels=1, num_repetitions=1, weights=torch.ones(1, 1, 1, 1))
    x = torch.randn(2, 1)

    with pytest.raises(UnsupportedOperationError):
        node.log_likelihood(x)
    with pytest.raises(UnsupportedOperationError):
        node.expectation_maximization(x)
    with pytest.raises(UnsupportedOperationError):
        node.maximum_likelihood_estimation(x)
    with pytest.raises(UnsupportedOperationError):
        node.marginalize([0])


def test_signed_sum_signed_eval_cache_hit_uses_cached_result():
    child = DummyInput(out_features=1, out_channels=2, num_repetitions=1, signed_eval_dim=4)
    weights = torch.tensor([[[[1.0]], [[2.0]]]])
    node = SignedSum(inputs=child, out_channels=1, num_repetitions=1, weights=weights)

    x = torch.randn(4, 1)
    cache = Cache()
    first_logabs, first_sign = node.signed_logabs_and_sign(x, cache=cache)
    second_logabs, second_sign = node.signed_logabs_and_sign(x, cache=cache)

    assert child.signed_calls == 1
    assert first_logabs is second_logabs
    assert first_sign is second_sign


def test_signed_sum_signed_eval_without_cache_and_bad_child_shape_raises():
    child = DummyInput(out_features=1, out_channels=2, num_repetitions=1, signed_eval_dim=4)
    node = SignedSum(inputs=child, out_channels=1, num_repetitions=1, weights=torch.ones(1, 2, 1, 1))

    x = torch.randn(3, 1)
    logabs, sign = node.signed_logabs_and_sign(x)
    assert logabs.shape == (3, 1, 1, 1)
    assert sign.shape == (3, 1, 1, 1)

    broken_child = DummyInput(out_features=1, out_channels=2, num_repetitions=1, signed_eval_dim=3)
    broken_node = SignedSum(
        inputs=broken_child, out_channels=1, num_repetitions=1, weights=torch.ones(1, 2, 1, 1)
    )

    with pytest.raises(ShapeError):
        broken_node.signed_logabs_and_sign(x)


def test_signed_sum_sample_rejects_evidence_and_negative_weights():
    child = DummyInput(out_features=1, out_channels=2, num_repetitions=1)
    pos_node = SignedSum(inputs=child, out_channels=1, num_repetitions=1, weights=torch.ones(1, 2, 1, 1))

    with pytest.raises(UnsupportedOperationError):
        pos_node.sample(data=torch.zeros(2, 1))

    neg_node = SignedSum(
        inputs=child,
        out_channels=1,
        num_repetitions=1,
        weights=torch.tensor([[[[-1.0]], [[1.0]]]]),
    )
    with pytest.raises(UnsupportedOperationError):
        neg_node.sample(data=torch.full((2, 1), torch.nan))


def test_signed_sum_sample_rejects_bad_weight_dim_and_repetitions():
    child = DummyInput(out_features=1, out_channels=2, num_repetitions=1)

    rep_node = SignedSum(inputs=child, out_channels=1, num_repetitions=2, weights=torch.ones(1, 2, 1, 2))
    with pytest.raises(UnsupportedOperationError):
        rep_node.sample(data=torch.full((2, 1), torch.nan))

    node = SignedSum(inputs=child, out_channels=1, num_repetitions=1, weights=torch.ones(1, 2, 1, 1))
    node.weights = nn.Parameter(torch.ones(1))
    with pytest.raises(ShapeError):
        node.sample(data=torch.full((2, 1), torch.nan))


@pytest.mark.parametrize("is_mpe", [True, False])
def test_signed_sum_sample_success_paths(is_mpe: bool):
    child = DummyInput(out_features=1, out_channels=2, num_repetitions=1)
    node = SignedSum(
        inputs=child,
        out_channels=1,
        num_repetitions=1,
        weights=torch.tensor([[[[1.0]], [[0.0]]]]),
    )

    sampled = node.sample(is_mpe=is_mpe)

    assert sampled.shape == (1, 1)
    assert torch.isfinite(sampled).all()
    assert child.sample_calls == 1
