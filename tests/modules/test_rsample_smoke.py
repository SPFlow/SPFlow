import numpy as np
import pytest
import torch

from spflow.meta.data.scope import Scope
from spflow.modules.conv.prod_conv import ProdConv
from spflow.modules.conv.sum_conv import SumConv
from spflow.modules.einsum import einsum_layer as einsum_layer_module
from spflow.modules.einsum.einsum_layer import EinsumLayer
from spflow.modules.einsum import linsum_layer as linsum_layer_module
from spflow.modules.einsum.linsum_layer import LinsumLayer
from spflow.modules.leaves.normal import Normal
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.ops.cat import Cat
from spflow.modules.ops.split_by_index import SplitByIndex
from spflow.modules.ops.split_consecutive import SplitConsecutive
from spflow.modules.ops.split_interleaved import SplitInterleaved
from spflow.modules.products.product import Product
from spflow.modules.sums import sum as sum_module
from spflow.modules.sums.elementwise_sum import ElementwiseSum
from spflow.modules.sums.repetition_mixing_layer import RepetitionMixingLayer
from spflow.modules.sums.signed_sum import SignedSum
from spflow.modules.sums.sum import Sum
from spflow.modules.wrapper.image_wrapper import ImageWrapper
from spflow.exceptions import UnsupportedOperationError
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from spflow.utils.sampling_context import init_default_sampling_context


class _ChannelEchoLeaf(Module):
    """Deterministic leaf that writes routed channel indices into sample tensor."""

    def __init__(self, scope: Scope, out_channels: int):
        super().__init__()
        self.scope = scope
        self.in_shape = ModuleShape(features=len(scope.query), channels=1, repetitions=1)
        self.out_shape = ModuleShape(features=len(scope.query), channels=out_channels, repetitions=1)

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([[Scope([rv])] for rv in self.scope.query], dtype=object)

    def log_likelihood(self, data, cache=None):
        del cache
        return torch.zeros(
            data.shape[0],
            self.out_shape.features,
            self.out_shape.channels,
            self.out_shape.repetitions,
            device=data.device,
        )

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None):
        del is_mpe, cache
        data = self._prepare_sample_data(num_samples, data)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])
        data[:, self.scope.query] = sampling_ctx.channel_index.to(data.dtype)
        return data

    def rsample(
        self,
        num_samples=None,
        data=None,
        is_mpe=False,
        cache=None,
        sampling_ctx=None,
        method="simple",
        tau=1.0,
        hard=True,
    ):
        del is_mpe, cache, method, tau, hard
        data = self._prepare_sample_data(num_samples, data)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])
        data[:, self.scope.query] = sampling_ctx.channel_index.to(data.dtype)
        return data

    def marginalize(self, marg_rvs, prune=True, cache: Cache | None = None):
        del marg_rvs, prune, cache
        return self


def _ctx(batch: int, features: int, channels: int) -> SamplingContext:
    return SamplingContext(
        channel_index=torch.zeros((batch, features), dtype=torch.long),
        mask=torch.ones((batch, features), dtype=torch.bool),
        repetition_index=torch.zeros((batch,), dtype=torch.long),
    )


def _fixed_indices(shape: tuple[int, ...], num_classes: int, device: torch.device) -> torch.Tensor:
    n = int(np.prod(shape))
    return (torch.arange(n, device=device) % num_classes).reshape(shape)


def _fake_categorical_sample(self, sample_shape=torch.Size()):
    assert sample_shape == torch.Size()
    source = self.logits if self.logits is not None else self.probs
    assert source is not None
    shape = tuple(source.shape[:-1])
    num_classes = source.shape[-1]
    return _fixed_indices(shape=shape, num_classes=num_classes, device=source.device)


def _fake_diff_selector(
    *,
    dim: int,
    is_mpe: bool,
    hard: bool = True,
    tau: float = 1.0,
    logits: torch.Tensor | None = None,
    log_weights: torch.Tensor | None = None,
    method=None,
):
    del is_mpe, hard, tau, method
    src = logits if logits is not None else log_weights
    assert src is not None
    if dim < 0:
        dim = src.ndim + dim
    out_shape = tuple(src.shape[:dim]) + tuple(src.shape[dim + 1 :])
    idx = _fixed_indices(shape=out_shape, num_classes=src.shape[dim], device=src.device)
    selector = torch.nn.functional.one_hot(idx, num_classes=src.shape[dim]).to(src.dtype)
    if dim != src.ndim - 1:
        selector = selector.movedim(-1, dim)
    return selector


def _assert_has_finite_grad(param: torch.Tensor) -> None:
    assert param.grad is not None
    assert torch.isfinite(param.grad).all()


def test_sum_rsample_has_gradient_on_logits():
    leaf = Normal(scope=Scope([0, 1, 2]), out_channels=4, num_repetitions=1)
    module = Sum(inputs=leaf, out_channels=3, num_repetitions=1)
    n = 6
    data = torch.full((n, 3), torch.nan)
    ctx = _ctx(batch=n, features=3, channels=3)

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    loss = out.mean()
    loss.backward()

    assert module.logits.grad is not None


def test_linsum_rsample_has_gradient_on_logits():
    left = Normal(scope=Scope([0, 1]), out_channels=3, num_repetitions=1)
    right = Normal(scope=Scope([2, 3]), out_channels=3, num_repetitions=1)
    module = LinsumLayer(inputs=[left, right], out_channels=2, num_repetitions=1)
    n = 5
    data = torch.full((n, 4), torch.nan)
    ctx = _ctx(batch=n, features=2, channels=2)

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    loss = out.mean()
    loss.backward()

    assert module.logits.grad is not None


def test_einsum_rsample_has_gradient_on_logits():
    left = Normal(scope=Scope([0, 1]), out_channels=2, num_repetitions=1)
    right = Normal(scope=Scope([2, 3]), out_channels=2, num_repetitions=1)
    module = EinsumLayer(inputs=[left, right], out_channels=2, num_repetitions=1)
    n = 4
    data = torch.full((n, 4), torch.nan)
    ctx = _ctx(batch=n, features=2, channels=2)

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    loss = out.mean()
    loss.backward()

    assert module.logits.grad is not None


def test_elementwise_sum_rsample_has_gradient_on_logits():
    a = Normal(scope=Scope([0, 1, 2]), out_channels=3, num_repetitions=1)
    b = Normal(scope=Scope([0, 1, 2]), out_channels=3, num_repetitions=1)
    module = ElementwiseSum(inputs=[a, b], out_channels=2, num_repetitions=1)
    n = 6
    data = torch.full((n, 3), torch.nan)
    ctx = _ctx(batch=n, features=3, channels=2)

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    loss = out.mean()
    loss.backward()

    assert module.logits.grad is not None


def test_repetition_mixing_layer_rsample_has_gradient_on_logits():
    leaf = Normal(scope=Scope([0]), out_channels=3, num_repetitions=2)
    module = RepetitionMixingLayer(inputs=leaf, out_channels=3, num_repetitions=2)
    n = 5
    data = torch.full((n, 1), torch.nan)
    ctx = SamplingContext(
        channel_index=torch.zeros((n, 1), dtype=torch.long),
        mask=torch.ones((n, 1), dtype=torch.bool),
        repetition_index=torch.zeros((n,), dtype=torch.long),
    )

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    loss = out.mean()
    loss.backward()

    _assert_has_finite_grad(module.logits)


def test_sum_conv_rsample_shape_and_finite():
    leaf = Normal(scope=Scope(list(range(16))), out_channels=3, num_repetitions=1)
    module = SumConv(inputs=leaf, out_channels=2, kernel_size=2, num_repetitions=1)
    n = 5
    data = torch.full((n, 16), torch.nan)
    ctx = _ctx(batch=n, features=16, channels=2)

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    assert out.shape == (n, 16)
    assert torch.isfinite(out).all()


def test_sum_conv_rsample_has_gradient_on_logits():
    leaf = Normal(scope=Scope(list(range(16))), out_channels=3, num_repetitions=1)
    module = SumConv(inputs=leaf, out_channels=2, kernel_size=2, num_repetitions=1)
    n = 5
    data = torch.full((n, 16), torch.nan)
    ctx = _ctx(batch=n, features=16, channels=2)

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    loss = out.mean()
    loss.backward()

    _assert_has_finite_grad(module.logits)


def test_signed_sum_rsample_has_gradient_on_weights_supported_case():
    leaf = Normal(scope=Scope([0, 1, 2]), out_channels=3, num_repetitions=1)
    weights = torch.rand(3, 3, 2, 1)
    module = SignedSum(inputs=leaf, out_channels=2, num_repetitions=1, weights=weights)
    n = 6
    data = torch.full((n, 3), torch.nan)
    ctx = _ctx(batch=n, features=3, channels=2)

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    loss = out.mean()
    loss.backward()

    _assert_has_finite_grad(module.weights)


def test_signed_sum_rsample_raises_for_conditional_evidence():
    leaf = Normal(scope=Scope([0, 1]), out_channels=2, num_repetitions=1)
    module = SignedSum(inputs=leaf, out_channels=1, num_repetitions=1, weights=torch.rand(2, 2, 1, 1))
    data = torch.zeros((3, 2))
    ctx = _ctx(batch=3, features=2, channels=1)

    with pytest.raises(UnsupportedOperationError):
        module.rsample(data=data, sampling_ctx=ctx, method="simple")


def test_signed_sum_rsample_raises_for_negative_weights():
    leaf = Normal(scope=Scope([0, 1]), out_channels=2, num_repetitions=1)
    weights = torch.tensor([[[[1.0]], [[-0.5]]], [[[0.2]], [[0.8]]]])
    module = SignedSum(inputs=leaf, out_channels=1, num_repetitions=1, weights=weights)
    data = torch.full((3, 2), torch.nan)
    ctx = _ctx(batch=3, features=2, channels=1)

    with pytest.raises(UnsupportedOperationError):
        module.rsample(data=data, sampling_ctx=ctx, method="simple")


def test_signed_sum_rsample_raises_for_multiple_repetitions():
    leaf = Normal(scope=Scope([0]), out_channels=2, num_repetitions=2)
    weights = torch.rand(1, 2, 1, 2)
    module = SignedSum(inputs=leaf, out_channels=1, num_repetitions=2, weights=weights)
    data = torch.full((3, 1), torch.nan)
    ctx = SamplingContext(
        channel_index=torch.zeros((3, 1), dtype=torch.long),
        mask=torch.ones((3, 1), dtype=torch.bool),
        repetition_index=torch.zeros((3,), dtype=torch.long),
    )

    with pytest.raises(UnsupportedOperationError):
        module.rsample(data=data, sampling_ctx=ctx, method="simple")


def test_product_rsample_propagates_gradient_to_descendant_params():
    leaf = Normal(scope=Scope([0, 1, 2]), out_channels=3, num_repetitions=1)
    inner = Sum(inputs=leaf, out_channels=2, num_repetitions=1)
    module = Product(inputs=inner)
    n = 7
    data = torch.full((n, 3), torch.nan)
    ctx = SamplingContext(
        channel_index=torch.zeros((n, 1), dtype=torch.long),
        mask=torch.ones((n, 1), dtype=torch.bool),
        repetition_index=torch.zeros((n,), dtype=torch.long),
    )

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    loss = out.mean()
    loss.backward()

    _assert_has_finite_grad(inner.logits)


def test_cat_rsample_propagates_gradient_to_child_params():
    a_leaf = Normal(scope=Scope([0, 1]), out_channels=3, num_repetitions=1)
    b_leaf = Normal(scope=Scope([2, 3]), out_channels=3, num_repetitions=1)
    a = Sum(inputs=a_leaf, out_channels=2, num_repetitions=1)
    b = Sum(inputs=b_leaf, out_channels=2, num_repetitions=1)
    module = Cat(inputs=[a, b], dim=1)
    n = 5
    data = torch.full((n, 4), torch.nan)
    ctx = _ctx(batch=n, features=4, channels=2)

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    loss = out.mean()
    loss.backward()

    _assert_has_finite_grad(a.logits)
    _assert_has_finite_grad(b.logits)


def test_split_rsample_propagates_gradient_to_child_params():
    leaf = Normal(scope=Scope([0, 1, 2, 3]), out_channels=3, num_repetitions=1)
    inner = Sum(inputs=leaf, out_channels=2, num_repetitions=1)
    module = SplitInterleaved(inputs=inner, num_splits=2, dim=1)
    n = 6
    data = torch.full((n, 4), torch.nan)
    ctx = _ctx(batch=n, features=4, channels=2)

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    loss = out.mean()
    loss.backward()

    _assert_has_finite_grad(inner.logits)


def test_split_consecutive_rsample_propagates_gradient_to_child_params():
    leaf = Normal(scope=Scope([0, 1, 2, 3]), out_channels=3, num_repetitions=1)
    inner = Sum(inputs=leaf, out_channels=2, num_repetitions=1)
    module = SplitConsecutive(inputs=inner, num_splits=2, dim=1)
    n = 6
    data = torch.full((n, 4), torch.nan)
    ctx = _ctx(batch=n, features=4, channels=2)

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    loss = out.mean()
    loss.backward()

    _assert_has_finite_grad(inner.logits)


def test_split_by_index_rsample_propagates_gradient_to_child_params():
    leaf = Normal(scope=Scope([0, 1, 2, 3]), out_channels=3, num_repetitions=1)
    inner = Sum(inputs=leaf, out_channels=2, num_repetitions=1)
    module = SplitByIndex(inputs=inner, indices=[[0, 2], [1, 3]])
    n = 6
    data = torch.full((n, 4), torch.nan)
    ctx = SamplingContext(
        channel_index=torch.zeros((n, 2), dtype=torch.long),
        mask=torch.ones((n, 2), dtype=torch.bool),
        repetition_index=torch.zeros((n,), dtype=torch.long),
    )

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    loss = out.mean()
    loss.backward()

    _assert_has_finite_grad(inner.logits)


def test_prod_conv_rsample_propagates_gradient_to_child_params():
    leaf = Normal(scope=Scope(list(range(16))), out_channels=3, num_repetitions=1)
    inner = Sum(inputs=leaf, out_channels=2, num_repetitions=1)
    module = ProdConv(inputs=inner, kernel_size_h=2, kernel_size_w=2)
    n = 5
    data = torch.full((n, 16), torch.nan)
    ctx = _ctx(batch=n, features=4, channels=2)

    out = module.rsample(data=data, sampling_ctx=ctx, method="simple")
    loss = out.mean()
    loss.backward()

    _assert_has_finite_grad(inner.logits)


def test_image_wrapper_rsample_propagates_gradient_to_wrapped_params():
    leaf = Normal(scope=Scope(list(range(16))), out_channels=3, num_repetitions=1)
    product = Product(inputs=leaf)
    root = Sum(inputs=product, out_channels=1, num_repetitions=1)
    module = ImageWrapper(module=root, num_channel=1, height=4, width=4)
    n = 5
    data = torch.full((n, 1, 4, 4), torch.nan)

    out = module.rsample(data=data, method="simple")
    loss = out.mean()
    loss.backward()

    _assert_has_finite_grad(root.logits)


def test_sum_sample_equals_rsample_with_patched_routing(monkeypatch):
    monkeypatch.setattr(torch.distributions.Categorical, "sample", _fake_categorical_sample)
    monkeypatch.setattr(sum_module, "sample_categorical_differentiably", _fake_diff_selector)

    leaf = _ChannelEchoLeaf(scope=Scope([0, 1, 2]), out_channels=4)
    module = Sum(inputs=leaf, out_channels=3, num_repetitions=1)

    n = 7
    data = torch.full((n, 3), torch.nan)
    ctx_sample = _ctx(batch=n, features=3, channels=3)
    ctx_rsample = _ctx(batch=n, features=3, channels=3)

    samples = module.sample(data=data.clone(), sampling_ctx=ctx_sample, is_mpe=False)
    rsamples = module.rsample(data=data.clone(), sampling_ctx=ctx_rsample, is_mpe=False, method="simple")

    torch.testing.assert_close(samples, rsamples, rtol=0.0, atol=0.0)


def test_linsum_sample_equals_rsample_with_patched_routing(monkeypatch):
    monkeypatch.setattr(torch.distributions.Categorical, "sample", _fake_categorical_sample)
    monkeypatch.setattr(linsum_layer_module, "sample_categorical_differentiably", _fake_diff_selector)

    left = _ChannelEchoLeaf(scope=Scope([0, 1]), out_channels=3)
    right = _ChannelEchoLeaf(scope=Scope([2, 3]), out_channels=3)
    module = LinsumLayer(inputs=[left, right], out_channels=2, num_repetitions=1)

    n = 5
    data = torch.full((n, 4), torch.nan)
    ctx_sample = _ctx(batch=n, features=2, channels=2)
    ctx_rsample = _ctx(batch=n, features=2, channels=2)

    samples = module.sample(data=data.clone(), sampling_ctx=ctx_sample, is_mpe=False)
    rsamples = module.rsample(data=data.clone(), sampling_ctx=ctx_rsample, is_mpe=False, method="simple")

    torch.testing.assert_close(samples, rsamples, rtol=0.0, atol=0.0)


def test_einsum_sample_equals_rsample_with_patched_routing(monkeypatch):
    monkeypatch.setattr(torch.distributions.Categorical, "sample", _fake_categorical_sample)
    monkeypatch.setattr(einsum_layer_module, "sample_categorical_differentiably", _fake_diff_selector)

    left = _ChannelEchoLeaf(scope=Scope([0, 1]), out_channels=2)
    right = _ChannelEchoLeaf(scope=Scope([2, 3]), out_channels=2)
    module = EinsumLayer(inputs=[left, right], out_channels=2, num_repetitions=1)

    n = 6
    data = torch.full((n, 4), torch.nan)
    ctx_sample = _ctx(batch=n, features=2, channels=2)
    ctx_rsample = _ctx(batch=n, features=2, channels=2)

    samples = module.sample(data=data.clone(), sampling_ctx=ctx_sample, is_mpe=False)
    rsamples = module.rsample(data=data.clone(), sampling_ctx=ctx_rsample, is_mpe=False, method="simple")

    torch.testing.assert_close(samples, rsamples, rtol=0.0, atol=0.0)
