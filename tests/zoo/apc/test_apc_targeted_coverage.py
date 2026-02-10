"""Targeted branch coverage tests for APC decoders, encoders, and model helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.meta.data.scope import Scope
from spflow.modules.conv.sum_conv import SumConv
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.sums.repetition_mixing_layer import RepetitionMixingLayer
from spflow.utils.sampling_context import SamplingContext
from spflow.zoo.apc.config import ApcConfig, ApcLossWeights
from spflow.zoo.apc.decoders import ConvDecoder2D, MLPDecoder1D, NeuralDecoder2D
from spflow.zoo.apc.encoders.base import LatentStats
from spflow.zoo.apc.encoders.convpc_joint_encoder import (
    _LatentFeaturePacking,
    _LatentSelectionCapture,
    _PairwiseLatentProduct,
    _default_normal_leaf,
    ConvPcJointEncoder,
)
from spflow.zoo.apc.encoders.einet_joint_encoder import EinetJointEncoder
from spflow.zoo.apc.model import AutoencodingPC


class _DummyModule(Module):
    def __init__(
        self,
        *,
        features: int,
        channels: int = 2,
        repetitions: int = 1,
        scope_query: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if scope_query is None:
            scope_query = tuple(range(features))
        self.scope = Scope(scope_query)
        self.in_shape = ModuleShape(features=features, channels=channels, repetitions=repetitions)
        self.out_shape = ModuleShape(features=features, channels=channels, repetitions=repetitions)
        self._feature_to_scope = np.empty((features, repetitions), dtype=object)
        for r in range(repetitions):
            for f in range(features):
                self._feature_to_scope[f, r] = Scope([scope_query[f % len(scope_query)]])

        self.force_bad_feature_count = False
        self.last_sampling_ctx: SamplingContext | None = None

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self._feature_to_scope

    def log_likelihood(self, data: torch.Tensor, cache=None) -> torch.Tensor:
        del cache
        b = data.shape[0]
        f = self.out_shape.features + (1 if self.force_bad_feature_count else 0)
        out = torch.zeros((b, f, self.out_shape.channels, self.out_shape.repetitions), device=data.device)
        if f >= 2:
            out[:, 0] = 1.0
            out[:, 1] = 2.0
        return out

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None) -> torch.Tensor:
        del num_samples, is_mpe, cache
        self.last_sampling_ctx = sampling_ctx
        assert data is not None
        return torch.full_like(data, 3.0)

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
    ) -> torch.Tensor:
        del num_samples, is_mpe, cache, method, tau, hard
        self.last_sampling_ctx = sampling_ctx
        assert data is not None
        return torch.full_like(data, 4.0)

    def marginalize(self, marg_rvs, prune=True, cache=None):
        del marg_rvs, prune, cache
        return None


class _StubEncoder:
    def __init__(self, *, num_x_features: int = 4, latent_dim: int = 2, bad_latent_out: bool = False) -> None:
        self.num_x_features = num_x_features
        self.latent_dim = latent_dim
        self.bad_latent_out = bad_latent_out
        self.last_tau_encode = None
        self.last_tau_decode = None
        self.last_tau_prior = None

    def encode(self, x, *, mpe=False, tau=1.0, return_latent_stats=False):
        del mpe
        self.last_tau_encode = tau
        b = x.shape[0]
        z = torch.full((b, self.latent_dim), 0.25, device=x.device)
        if not return_latent_stats:
            return z
        if self.bad_latent_out:
            return z
        stats = LatentStats(mu=torch.zeros_like(z), logvar=torch.zeros_like(z))
        return stats, z

    def decode(self, z, *, x=None, mpe=False, tau=1.0, fill_evidence=False):
        del mpe, fill_evidence
        self.last_tau_decode = tau
        b = z.shape[0]
        out = torch.zeros((b, self.num_x_features), device=z.device)
        if x is not None:
            out = torch.nan_to_num(x.reshape(b, -1), nan=0.0).to(out.dtype)
        return out

    def joint_log_likelihood(self, x, z):
        del z
        return torch.zeros((x.shape[0],), device=x.device)

    def log_likelihood_x(self, x):
        return torch.zeros((x.shape[0],), device=x.device)

    def sample_prior_z(self, num_samples, *, tau=1.0):
        self.last_tau_prior = tau
        return torch.zeros((num_samples, self.latent_dim))

    def latent_stats(self, x, *, tau=1.0):
        del x, tau
        return LatentStats(mu=torch.zeros(1, self.latent_dim), logvar=torch.zeros(1, self.latent_dim))


def test_mlp_decoder_validation_and_forward_branches():
    with pytest.raises(InvalidParameterError, match="latent_dim"):
        MLPDecoder1D(latent_dim=0, output_dim=4)
    with pytest.raises(InvalidParameterError, match="output_dim"):
        MLPDecoder1D(latent_dim=2, output_dim=0)
    with pytest.raises(InvalidParameterError, match="hidden_dims must contain"):
        MLPDecoder1D(latent_dim=2, output_dim=4, hidden_dims=())
    with pytest.raises(InvalidParameterError, match="hidden_dims must be positive"):
        MLPDecoder1D(latent_dim=2, output_dim=4, hidden_dims=(8, 0))
    with pytest.raises(InvalidParameterError, match="out_activation"):
        MLPDecoder1D(latent_dim=2, output_dim=4, hidden_dims=(8,), out_activation="bad")

    tanh_dec = MLPDecoder1D(latent_dim=2, output_dim=3, hidden_dims=(8,), out_activation="tanh")
    sig_dec = MLPDecoder1D(latent_dim=2, output_dim=3, hidden_dims=(8,), out_activation="sigmoid")
    z = torch.randn(4, 2)
    assert tanh_dec(z).shape == (4, 3)
    assert sig_dec(z).shape == (4, 3)

    with pytest.raises(InvalidParameterError, match="Expected latent feature size"):
        tanh_dec(torch.randn(4, 3))


def test_conv_decoder2d_validation_and_interpolation_branch():
    with pytest.raises(InvalidParameterError, match="latent_dim"):
        ConvDecoder2D(latent_dim=0, output_shape=(1, 8, 8))
    with pytest.raises(InvalidParameterError, match="output_shape"):
        ConvDecoder2D(latent_dim=2, output_shape=(1, 8))
    with pytest.raises(InvalidParameterError, match="entries must be positive"):
        ConvDecoder2D(latent_dim=2, output_shape=(1, 0, 8))
    with pytest.raises(InvalidParameterError, match="base_channels"):
        ConvDecoder2D(latent_dim=2, output_shape=(1, 8, 8), base_channels=0)
    with pytest.raises(InvalidParameterError, match="num_upsamples"):
        ConvDecoder2D(latent_dim=2, output_shape=(1, 8, 8), num_upsamples=-1)
    with pytest.raises(InvalidParameterError, match="out_activation"):
        ConvDecoder2D(latent_dim=2, output_shape=(1, 8, 8), out_activation="bad")

    dec = ConvDecoder2D(
        latent_dim=3,
        output_shape=(1, 7, 5),
        base_channels=8,
        num_upsamples=2,
        out_activation="tanh",
    )
    x = dec(torch.randn(2, 3))
    assert x.shape == (2, 1, 7, 5)

    with pytest.raises(InvalidParameterError, match="Expected latent feature size"):
        dec(torch.randn(2, 4))


def test_neural_decoder2d_validation_and_scale_loop():
    with pytest.raises(InvalidParameterError, match="latent_dim"):
        NeuralDecoder2D(latent_dim=0, output_shape=(1, 32, 32))
    with pytest.raises(InvalidParameterError, match="output_shape"):
        NeuralDecoder2D(latent_dim=2, output_shape=(1, 32))
    with pytest.raises(InvalidParameterError, match="entries must be positive"):
        NeuralDecoder2D(latent_dim=2, output_shape=(1, 0, 32))
    with pytest.raises(InvalidParameterError, match="must be >= 1"):
        NeuralDecoder2D(latent_dim=2, output_shape=(1, 32, 32), num_hidden=0)
    with pytest.raises(InvalidParameterError, match="num_scales"):
        NeuralDecoder2D(latent_dim=2, output_shape=(1, 32, 32), num_scales=1)
    with pytest.raises(InvalidParameterError, match="divisible"):
        NeuralDecoder2D(latent_dim=2, output_shape=(1, 30, 32), num_scales=2)
    with pytest.raises(InvalidParameterError, match="must be even"):
        NeuralDecoder2D(latent_dim=2, output_shape=(1, 32, 32), num_hidden=63)
    with pytest.raises(InvalidParameterError, match="out_activation"):
        NeuralDecoder2D(latent_dim=2, output_shape=(1, 32, 32), out_activation="bad")

    dec_identity = NeuralDecoder2D(
        latent_dim=4,
        output_shape=(1, 32, 32),
        num_hidden=64,
        num_res_hidden=16,
        num_res_layers=2,
        num_scales=3,
        out_activation="identity",
    )
    dec_sigmoid = NeuralDecoder2D(
        latent_dim=4,
        output_shape=(1, 32, 32),
        num_hidden=64,
        num_res_hidden=16,
        num_res_layers=2,
        num_scales=3,
        out_activation="sigmoid",
    )
    z = torch.randn(2, 4)
    assert dec_identity(z).shape == (2, 1, 32, 32)
    assert dec_sigmoid(z).shape == (2, 1, 32, 32)

    with pytest.raises(InvalidParameterError, match="Expected latent feature size"):
        dec_identity(torch.randn(2, 3))


def test_apc_model_cover_decode_without_decoder_and_misc_helpers():
    cfg = ApcConfig(latent_dim=2, rec_loss="mse", sample_tau=0.7, loss_weights=ApcLossWeights())
    stub = _StubEncoder()
    model = AutoencodingPC(encoder=stub, decoder=None, config=cfg)

    x = torch.randn(3, 4)
    z = model.encode(x)
    assert z.shape == (3, 2)
    assert stub.last_tau_encode == pytest.approx(0.7)

    x_rec = model.decode(z)
    assert x_rec.shape == (3, 4)
    assert stub.last_tau_decode == pytest.approx(0.7)

    assert model.reconstruct(x).shape == (3, 4)
    assert model.sample_z(5).shape == (5, 2)
    assert model.sample_x(5).shape == (5, 4)
    assert stub.last_tau_prior == pytest.approx(0.7)

    assert model.log_likelihood_x(x).shape == (3,)
    assert model.joint_log_likelihood(x, z).shape == (3,)
    assert set(model.forward(x).keys()) >= {"total", "rec", "kld", "nll"}
    assert "latent_dim=2" in model.extra_repr()


def test_apc_model_error_branches_for_invalid_shapes_and_stats():
    with pytest.raises(InvalidParameterError, match="n_bits"):
        ApcConfig(latent_dim=2, n_bits=1, rec_loss="mse", sample_tau=1.0, loss_weights=ApcLossWeights())

    cfg = ApcConfig(latent_dim=2, rec_loss="mse", sample_tau=1.0, loss_weights=ApcLossWeights())
    model = AutoencodingPC(encoder=_StubEncoder(), decoder=nn.Identity(), config=cfg)

    with pytest.raises(InvalidParameterError, match="at least one feature axis"):
        model._flatten_tensor(torch.randn(3))

    with pytest.raises(InvalidParameterError, match="shape mismatch"):
        model._reconstruction_loss(torch.randn(2, 4), torch.randn(2, 3))

    rec_with_nans = model._reconstruction_loss(torch.full((2, 4), float("nan")), torch.randn(2, 4))
    assert torch.isnan(rec_with_nans)

    model_bce = AutoencodingPC(
        encoder=_StubEncoder(),
        decoder=nn.Identity(),
        config=ApcConfig(latent_dim=2, rec_loss="bce", sample_tau=1.0, loss_weights=ApcLossWeights()),
    )
    with pytest.raises(RuntimeError):
        model_bce._reconstruction_loss(torch.rand(2, 4), torch.randn(2, 4) * 10.0)

    model_bad_rec = AutoencodingPC(
        encoder=_StubEncoder(),
        decoder=nn.Identity(),
        config=ApcConfig(
            latent_dim=2, rec_loss="unsupported", sample_tau=1.0, loss_weights=ApcLossWeights()
        ),
    )
    with pytest.raises(InvalidParameterError, match="Unsupported rec_loss"):
        model_bad_rec._reconstruction_loss(torch.randn(2, 4), torch.randn(2, 4))

    with pytest.raises(InvalidParameterError, match="shape mismatch"):
        model._kld_from_stats(LatentStats(mu=torch.zeros(2, 2), logvar=torch.zeros(2, 3)))
    with pytest.raises(InvalidParameterError, match="at least one latent dimension"):
        model._kld_from_stats(LatentStats(mu=torch.zeros(2), logvar=torch.zeros(2)))

    bad_model = AutoencodingPC(encoder=_StubEncoder(bad_latent_out=True), decoder=nn.Identity(), config=cfg)
    with pytest.raises(InvalidParameterError, match=r"must return \(LatentStats, z\)"):
        bad_model.loss_components(torch.randn(2, 4))

    class _BadStatsEncoder(_StubEncoder):
        def encode(self, x, *, mpe=False, tau=1.0, return_latent_stats=False):
            if return_latent_stats:
                return torch.zeros(x.shape[0], self.latent_dim), torch.zeros(x.shape[0], self.latent_dim)
            return super().encode(x, mpe=mpe, tau=tau, return_latent_stats=return_latent_stats)

    with pytest.raises(InvalidParameterError, match="invalid latent stats type"):
        AutoencodingPC(encoder=_BadStatsEncoder(), decoder=nn.Identity(), config=cfg).loss_components(
            torch.randn(2, 4)
        )


def test_einet_encoder_validation_and_helper_errors():
    with pytest.raises(InvalidParameterError, match="num_x_features"):
        EinetJointEncoder(num_x_features=0, latent_dim=2)
    with pytest.raises(InvalidParameterError, match="latent_dim"):
        EinetJointEncoder(num_x_features=4, latent_dim=0)
    with pytest.raises(InvalidParameterError, match="posterior_stat_samples"):
        EinetJointEncoder(num_x_features=4, latent_dim=2, posterior_stat_samples=0)
    with pytest.raises(InvalidParameterError, match="posterior_var_floor"):
        EinetJointEncoder(num_x_features=4, latent_dim=2, posterior_var_floor=0.0)
    with pytest.raises(InvalidParameterError, match="structure='top-down'"):
        EinetJointEncoder(num_x_features=4, latent_dim=2, structure="bottom-up")

    with pytest.raises(InvalidParameterError, match="LeafModule"):
        EinetJointEncoder(
            num_x_features=4,
            latent_dim=2,
            x_leaf_factory=lambda *_: object(),
        )

    def _wrong_scope_leaf(scope_indices, out_channels, num_repetitions):
        del scope_indices
        from spflow.modules.leaves import Normal

        return Normal(scope=[999], out_channels=out_channels, num_repetitions=num_repetitions)

    with pytest.raises(InvalidParameterError, match="expected scope"):
        EinetJointEncoder(num_x_features=4, latent_dim=2, x_leaf_factory=_wrong_scope_leaf)

    enc = EinetJointEncoder(num_x_features=4, latent_dim=2, num_sums=3, num_leaves=3, depth=1, num_repetitions=2)
    with pytest.raises(ShapeError, match="at least 2 dimensions"):
        enc._flatten_x(torch.randn(4))
    with pytest.raises(ShapeError, match="flattened features"):
        enc._flatten_x(torch.randn(2, 3))
    with pytest.raises(ShapeError, match="at least 2 dimensions"):
        enc._flatten_z(torch.randn(4))
    with pytest.raises(ShapeError, match="latent_dim"):
        enc._flatten_z(torch.randn(2, 3))

    with pytest.raises(InvalidParameterError, match="num_samples must be provided"):
        enc._build_evidence(x_flat=None, z_flat=None)
    with pytest.raises(ShapeError, match="batch sizes must match"):
        enc._build_evidence(x_flat=torch.randn(2, 4), z_flat=torch.randn(3, 2))

    evidence = enc._build_evidence(x_flat=None, z_flat=None, num_samples=3)
    assert evidence.shape == (3, 6)
    assert torch.isnan(evidence).all()

    with pytest.raises(ShapeError, match="batch dimension"):
        enc._flatten_ll(torch.tensor(1.0))
    with pytest.raises(ShapeError, match="scalar log-likelihood"):
        enc._flatten_ll(torch.randn(2, 2))

    x = torch.randn(2, 4)
    assert enc.decode(enc.encode(x), mpe=True).shape == (2, 4)
    with pytest.raises(InvalidParameterError, match="num_samples must be >= 1"):
        enc.sample_prior_z(0)


def test_einet_latent_stats_branches_with_custom_sampling_contexts():
    enc = EinetJointEncoder(num_x_features=4, latent_dim=2, num_sums=3, num_leaves=3, depth=1, num_repetitions=2)
    b = 3

    ctx_latent = SamplingContext(
        channel_index=torch.zeros((b, enc.latent_dim), dtype=torch.long),
        mask=torch.ones((b, enc.latent_dim), dtype=torch.bool),
        repetition_index=torch.zeros((b, 1), dtype=torch.long),
    )
    stats1 = enc._latent_stats_from_leaf_params(ctx_latent, batch_size=b)
    assert stats1.mu.shape == (b, enc.latent_dim)

    ctx_full = SamplingContext(
        channel_index=torch.zeros((b, enc.num_x_features + enc.latent_dim), dtype=torch.long),
        mask=torch.ones((b, enc.num_x_features + enc.latent_dim), dtype=torch.bool),
        repetition_index=torch.zeros((b, enc.num_x_features + enc.latent_dim), dtype=torch.long),
    )
    stats2 = enc._latent_stats_from_leaf_params(ctx_full, batch_size=b)
    assert stats2.logvar.shape == (b, enc.latent_dim)

    ctx_fallback_rep = SamplingContext(
        channel_index=torch.zeros((b, 1), dtype=torch.long),
        mask=torch.ones((b, 1), dtype=torch.bool),
        repetition_index=torch.zeros((b, 5), dtype=torch.long),
    )
    stats3 = enc._latent_stats_from_leaf_params(ctx_fallback_rep, batch_size=b)
    assert stats3.mu.shape == (b, enc.latent_dim)

    bad_ctx = SamplingContext(
        channel_index=torch.zeros((b, 3), dtype=torch.long),
        mask=torch.ones((b, 3), dtype=torch.bool),
    )
    with pytest.raises(InvalidParameterError, match="Unexpected sampling context feature width"):
        enc._latent_stats_from_leaf_params(bad_ctx, batch_size=b)


def test_convpc_helper_modules_branches():
    child = _DummyModule(features=4, channels=2, repetitions=1)

    with pytest.raises(InvalidParameterError, match="even feature count"):
        _PairwiseLatentProduct(inputs=_DummyModule(features=3))

    pair = _PairwiseLatentProduct(inputs=child)
    ll = pair.log_likelihood(torch.randn(2, 4))
    assert ll.shape == (2, 2, 2, 1)

    child.force_bad_feature_count = True
    with pytest.raises(ShapeError, match="Unexpected feature size"):
        pair.log_likelihood(torch.randn(2, 4))
    child.force_bad_feature_count = False

    ctx_out = SamplingContext(
        channel_index=torch.zeros((2, pair.out_shape.features), dtype=torch.long),
        mask=torch.ones((2, pair.out_shape.features), dtype=torch.bool),
    )
    pair.sample(data=torch.randn(2, 4), sampling_ctx=ctx_out)
    assert child.last_sampling_ctx is ctx_out
    assert child.last_sampling_ctx.channel_index.shape[1] == pair.in_shape.features

    pair.rsample(data=torch.randn(2, 4), sampling_ctx=ctx_out)
    assert child.last_sampling_ctx.channel_index.shape[1] == pair.in_shape.features

    with pytest.raises(ShapeError, match="feature dimension mismatch"):
        bad_ctx = SamplingContext(
            channel_index=torch.zeros((2, 3), dtype=torch.long),
            mask=torch.ones((2, 3), dtype=torch.bool),
        )
        pair.sample(data=torch.randn(2, 4), sampling_ctx=bad_ctx)

    with pytest.raises(NotImplementedError, match="not implemented"):
        pair.marginalize([])

    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    perm_inv = torch.argsort(perm)
    pack = _LatentFeaturePacking(inputs=_DummyModule(features=2), target_features=4, perm=perm, perm_inv=perm_inv)
    packed_ll = pack.log_likelihood(torch.randn(2, 2))
    assert packed_ll.shape == (2, 4, 2, 1)

    with pytest.raises(InvalidParameterError, match="target_features"):
        _LatentFeaturePacking(inputs=_DummyModule(features=3), target_features=2, perm=None, perm_inv=None)
    with pytest.raises(InvalidParameterError, match="both be set"):
        _LatentFeaturePacking(inputs=_DummyModule(features=2), target_features=4, perm=perm, perm_inv=None)
    with pytest.raises(InvalidParameterError, match="perm length"):
        _LatentFeaturePacking(
            inputs=_DummyModule(features=2),
            target_features=4,
            perm=torch.tensor([0, 1], dtype=torch.long),
            perm_inv=torch.tensor([0, 1], dtype=torch.long),
        )

    ctx_target = SamplingContext(
        channel_index=torch.zeros((2, 4), dtype=torch.long),
        mask=torch.ones((2, 4), dtype=torch.bool),
    )
    pack.sample(data=torch.randn(2, 2), sampling_ctx=ctx_target)
    assert pack.inputs.last_sampling_ctx.channel_index.shape[1] == 2
    pack.rsample(data=torch.randn(2, 2), sampling_ctx=ctx_target)
    assert pack.inputs.last_sampling_ctx.channel_index.shape[1] == 2

    with pytest.raises(ShapeError, match="feature dimension mismatch"):
        bad_ctx2 = SamplingContext(
            channel_index=torch.zeros((2, 3), dtype=torch.long),
            mask=torch.ones((2, 3), dtype=torch.bool),
        )
        pack.rsample(data=torch.randn(2, 2), sampling_ctx=bad_ctx2)

    with pytest.raises(NotImplementedError, match="not implemented"):
        pack.marginalize([])

    captures: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def _capture_fn(ch, rep):
        captures.append((ch, rep))

    select = _LatentSelectionCapture(
        inputs=_DummyModule(features=2, scope_query=(1, 3)),
        capture_fn=_capture_fn,
    )

    ctx_data = SamplingContext(
        channel_index=torch.tensor([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]], dtype=torch.long),
        mask=torch.ones((2, 5), dtype=torch.bool),
        repetition_index=torch.tensor([[9, 8, 7, 6, 5], [4, 3, 2, 1, 0]], dtype=torch.long),
    )
    select.sample(data=torch.randn(2, 5), sampling_ctx=ctx_data)
    ch0, rep0 = captures[-1]
    assert torch.equal(ch0, torch.tensor([[6, 8], [1, 3]], dtype=torch.long))
    assert rep0 is not None and rep0.shape == (2, 2)

    ctx_fallback = SamplingContext(
        channel_index=torch.zeros((2, 4), dtype=torch.long),
        mask=torch.ones((2, 4), dtype=torch.bool),
        repetition_index=torch.zeros((2, 4), dtype=torch.long),
    )
    select.rsample(data=torch.randn(2, 7), sampling_ctx=ctx_fallback)
    ch1, rep1 = captures[-1]
    assert ch1.shape == (2, 2)
    assert rep1 is not None and rep1.shape == (2, 2)

    with pytest.raises(NotImplementedError, match="not implemented"):
        select.marginalize([])


def test_convpc_encoder_validation_and_rsample_error_propagation(monkeypatch):
    bad_kwargs = [
        dict(input_height=0),
        dict(input_channels=0),
        dict(latent_dim=0),
        dict(channels=0),
        dict(depth=0),
        dict(kernel_size=0),
        dict(num_repetitions=0),
        dict(architecture="bad"),
        dict(architecture="reference", depth=1),
        dict(architecture="reference", depth=2, latent_depth=1),
        dict(architecture="legacy", latent_depth=-1),
        dict(posterior_stat_samples=0),
        dict(posterior_var_floor=0.0),
        dict(x_leaf_channels=0),
    ]
    base_kwargs = dict(
        input_height=4,
        input_width=4,
        input_channels=1,
        latent_dim=4,
        channels=4,
        depth=2,
        kernel_size=2,
        num_repetitions=1,
        use_sum_conv=False,
        latent_depth=0,
    )
    for kwargs in bad_kwargs:
        with pytest.raises(InvalidParameterError):
            ConvPcJointEncoder(**(base_kwargs | kwargs))

    enc = ConvPcJointEncoder(
        input_height=4,
        input_width=4,
        input_channels=1,
        latent_dim=4,
        channels=4,
        depth=2,
        kernel_size=2,
        num_repetitions=1,
        use_sum_conv=False,
        latent_depth=0,
        architecture="reference",
    )

    with pytest.raises(ShapeError, match="rank-2 or rank-4"):
        enc._flatten_x(torch.randn(4))
    with pytest.raises(ShapeError, match="image shape mismatch"):
        enc._flatten_x(torch.randn(2, 1, 4, 3))

    x2 = torch.randn(2, 16)
    assert enc._flatten_x(x2).shape == (2, 16)
    assert enc._reshape_x_like(torch.randn(2, 16), x2).shape == (2, 16)

    x = torch.randn(2, 1, 4, 4)

    def _raise_rsample(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("force rsample failure")

    monkeypatch.setattr(enc.pc, "rsample", _raise_rsample)
    with pytest.raises(RuntimeError, match="force rsample failure"):
        enc.encode(x)

    z_in = torch.randn(2, 4)
    with pytest.raises(RuntimeError, match="force rsample failure"):
        enc.decode(z_in)

    with pytest.raises(RuntimeError, match="force rsample failure"):
        enc.sample_prior_z(2)


def test_convpc_internal_helper_remaining_branches():
    child = _DummyModule(features=4, channels=2, repetitions=1)
    pair = _PairwiseLatentProduct(inputs=child)

    ctx_in = SamplingContext(
        channel_index=torch.zeros((2, 4), dtype=torch.long),
        mask=torch.ones((2, 4), dtype=torch.bool),
    )
    pair.sample(data=torch.randn(2, 4), sampling_ctx=ctx_in)
    assert child.last_sampling_ctx is ctx_in

    ctx_one = SamplingContext(
        channel_index=torch.zeros((2, 1), dtype=torch.long),
        mask=torch.ones((2, 1), dtype=torch.bool),
    )
    pair.sample(data=torch.randn(2, 4), sampling_ctx=ctx_one)
    assert child.last_sampling_ctx.channel_index.shape[1] == 4
    ctx_one_rsample = SamplingContext(
        channel_index=torch.zeros((2, 1), dtype=torch.long),
        mask=torch.ones((2, 1), dtype=torch.bool),
    )
    pair.rsample(data=torch.randn(2, 4), sampling_ctx=ctx_one_rsample)
    assert child.last_sampling_ctx.channel_index.shape[1] == 4

    ctx_out_rsample = SamplingContext(
        channel_index=torch.zeros((2, 2), dtype=torch.long),
        mask=torch.ones((2, 2), dtype=torch.bool),
    )
    pair.rsample(data=torch.randn(2, 4), sampling_ctx=ctx_out_rsample)
    assert child.last_sampling_ctx.channel_index.shape[1] == 4

    with pytest.raises(ShapeError, match="feature dimension mismatch"):
        pair.rsample(
            data=torch.randn(2, 4),
            sampling_ctx=SamplingContext(
                channel_index=torch.zeros((2, 3), dtype=torch.long),
                mask=torch.ones((2, 3), dtype=torch.bool),
            ),
        )

    pack_child = _DummyModule(features=2, channels=2, repetitions=1)
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    perm_inv = torch.argsort(perm)
    pack = _LatentFeaturePacking(inputs=pack_child, target_features=4, perm=perm, perm_inv=perm_inv)

    pack_ctx_in = SamplingContext(
        channel_index=torch.zeros((2, 2), dtype=torch.long),
        mask=torch.ones((2, 2), dtype=torch.bool),
    )
    pack.sample(data=torch.randn(2, 2), sampling_ctx=pack_ctx_in)
    assert pack_child.last_sampling_ctx is pack_ctx_in

    pack_ctx_one = SamplingContext(
        channel_index=torch.zeros((2, 1), dtype=torch.long),
        mask=torch.ones((2, 1), dtype=torch.bool),
    )
    pack.sample(data=torch.randn(2, 2), sampling_ctx=pack_ctx_one)
    assert pack_child.last_sampling_ctx.channel_index.shape[1] == 2
    pack_ctx_one_rsample = SamplingContext(
        channel_index=torch.zeros((2, 1), dtype=torch.long),
        mask=torch.ones((2, 1), dtype=torch.bool),
    )
    pack.rsample(data=torch.randn(2, 2), sampling_ctx=pack_ctx_one_rsample)
    assert pack_child.last_sampling_ctx.channel_index.shape[1] == 2

    pack_child.force_bad_feature_count = True
    with pytest.raises(ShapeError, match="Unexpected feature size"):
        pack.log_likelihood(torch.randn(2, 2))
    pack_child.force_bad_feature_count = False

    with pytest.raises(ShapeError, match="feature dimension mismatch"):
        pack.sample(
            data=torch.randn(2, 2),
            sampling_ctx=SamplingContext(
                channel_index=torch.zeros((2, 3), dtype=torch.long),
                mask=torch.ones((2, 3), dtype=torch.bool),
            ),
        )

    captures = []
    select = _LatentSelectionCapture(
        inputs=_DummyModule(features=2, scope_query=(1, 3)),
        capture_fn=lambda ch, rep: captures.append((ch, rep)),
    )
    select.log_likelihood(torch.randn(2, 2))
    ctx_select = SamplingContext(
        channel_index=torch.zeros((2, 1), dtype=torch.long),
        mask=torch.ones((2, 1), dtype=torch.bool),
        repetition_index=torch.zeros((2, 1), dtype=torch.long),
    )
    select.sample(data=torch.randn(2, 2), sampling_ctx=ctx_select)
    _, rep = captures[-1]
    assert rep is not None and rep.shape == (2, 2)

    ctx_select_target = SamplingContext(
        channel_index=torch.zeros((2, 2), dtype=torch.long),
        mask=torch.ones((2, 2), dtype=torch.bool),
        repetition_index=torch.zeros((2, 2), dtype=torch.long),
    )
    select.rsample(data=torch.randn(2, 2), sampling_ctx=ctx_select_target)
    _, rep2 = captures[-1]
    assert rep2 is not None and rep2.shape == (2, 2)


def test_convpc_encoder_legacy_and_reference_additional_architecture_paths():
    legacy = ConvPcJointEncoder(
        input_height=4,
        input_width=4,
        input_channels=1,
        latent_dim=16,
        channels=4,
        depth=2,
        kernel_size=2,
        num_repetitions=2,
        use_sum_conv=True,
        latent_depth=0,
        architecture="legacy",
        latent_channels=2,
    )
    assert any(isinstance(m, SumConv) for m in legacy.layers)
    assert any(isinstance(m, RepetitionMixingLayer) for m in legacy.layers)
    assert legacy.latent_sum_layer is None

    x = torch.randn(2, 1, 4, 4)
    assert legacy.encode(x).shape == (2, 16)

    reference = ConvPcJointEncoder(
        input_height=4,
        input_width=4,
        input_channels=1,
        latent_dim=16,
        channels=4,
        depth=3,
        kernel_size=2,
        num_repetitions=2,
        use_sum_conv=True,
        latent_depth=0,
        architecture="reference",
    )
    assert any(isinstance(m, SumConv) for m in reference.layers)
    assert any(isinstance(m, RepetitionMixingLayer) for m in reference.layers)

    with pytest.raises(InvalidParameterError, match="even latent width"):
        ConvPcJointEncoder(
            input_height=4,
            input_width=4,
            input_channels=1,
            latent_dim=5,
            channels=4,
            depth=2,
            kernel_size=2,
            num_repetitions=1,
            use_sum_conv=False,
            latent_depth=0,
            architecture="reference",
        )


def test_convpc_encoder_helper_error_paths_and_latent_index_resolvers():
    enc = ConvPcJointEncoder(
        input_height=4,
        input_width=4,
        input_channels=1,
        latent_dim=4,
        channels=4,
        depth=2,
        kernel_size=2,
        num_repetitions=1,
        use_sum_conv=False,
        latent_depth=0,
        architecture="reference",
    )

    with pytest.raises(ShapeError, match="flattened features"):
        enc._flatten_x(torch.randn(2, 15))
    with pytest.raises(ShapeError, match="at least 2 dimensions"):
        enc._flatten_z(torch.randn(4))
    with pytest.raises(ShapeError, match="latent_dim"):
        enc._flatten_z(torch.randn(2, 5))

    with pytest.raises(InvalidParameterError, match="num_samples must be provided"):
        enc._build_evidence(x_flat=None, z_flat=None)
    with pytest.raises(ShapeError, match="batch sizes must match"):
        enc._build_evidence(x_flat=torch.randn(2, 16), z_flat=torch.randn(3, 4))
    evidence = enc._build_evidence(x_flat=None, z_flat=None, num_samples=2)
    assert evidence.shape == (2, 20)

    with pytest.raises(ShapeError, match="batch dimension"):
        enc._flatten_ll(torch.tensor(1.0))
    with pytest.raises(ShapeError, match="scalar log-likelihood"):
        enc._flatten_ll(torch.randn(2, 2))

    x = torch.randn(2, 1, 4, 4)
    z, _ = enc._posterior_sample(enc._flatten_x(x), mpe=True, tau=1.0, return_sampling_ctx=True)
    assert z.shape == (2, 4)

    loc = torch.zeros((enc.latent_dim, 3, 2))
    enc._last_latent_leaf_channel_index = torch.zeros((1, 1), dtype=torch.long)
    with pytest.raises(ShapeError, match="batch mismatch"):
        enc._resolve_latent_channel_indices(
            sampling_ctx=SamplingContext(num_samples=2),
            batch_size=2,
            loc=loc,
        )

    enc._last_latent_leaf_channel_index = torch.zeros((2, 2), dtype=torch.long)
    with pytest.raises(ShapeError, match="feature mismatch"):
        enc._resolve_latent_channel_indices(
            sampling_ctx=SamplingContext(num_samples=2),
            batch_size=2,
            loc=loc,
        )

    enc._last_latent_leaf_channel_index = torch.zeros((2, 1), dtype=torch.long)
    resolved = enc._resolve_latent_channel_indices(
        sampling_ctx=SamplingContext(num_samples=2),
        batch_size=2,
        loc=loc,
    )
    assert resolved.shape == (2, enc.latent_dim)
    enc._last_latent_leaf_channel_index = None

    for width in (1, enc.latent_dim, enc.num_x_features + enc.latent_dim):
        ctx = SamplingContext(
            channel_index=torch.zeros((2, width), dtype=torch.long),
            mask=torch.ones((2, width), dtype=torch.bool),
        )
        out = enc._resolve_latent_channel_indices(sampling_ctx=ctx, batch_size=2, loc=loc)
        assert out.shape == (2, enc.latent_dim)

    bad_ctx = SamplingContext(
        channel_index=torch.zeros((2, 3), dtype=torch.long),
        mask=torch.ones((2, 3), dtype=torch.bool),
    )
    with pytest.raises(InvalidParameterError, match="Unexpected sampling context feature width"):
        enc._resolve_latent_channel_indices(sampling_ctx=bad_ctx, batch_size=2, loc=loc)

    enc._last_latent_leaf_repetition_index = torch.zeros((1, 1), dtype=torch.long)
    with pytest.raises(ShapeError, match="batch mismatch"):
        enc._resolve_latent_repetition_indices(
            sampling_ctx=SamplingContext(num_samples=2),
            batch_size=2,
            loc=loc,
        )

    enc._last_latent_leaf_repetition_index = torch.zeros((2, 2), dtype=torch.long)
    with pytest.raises(ShapeError, match="feature mismatch"):
        enc._resolve_latent_repetition_indices(
            sampling_ctx=SamplingContext(num_samples=2),
            batch_size=2,
            loc=loc,
        )

    enc._last_latent_leaf_repetition_index = torch.zeros((2, 1), dtype=torch.long)
    rep_resolved = enc._resolve_latent_repetition_indices(
        sampling_ctx=SamplingContext(num_samples=2),
        batch_size=2,
        loc=loc,
    )
    assert rep_resolved.shape == (2, enc.latent_dim)
    enc._last_latent_leaf_repetition_index = None

    for rep in (
        None,
        torch.zeros((2,), dtype=torch.long),
        torch.zeros((2, 1), dtype=torch.long),
        torch.zeros((2, enc.latent_dim), dtype=torch.long),
        torch.zeros((2, enc.num_x_features + enc.latent_dim), dtype=torch.long),
        torch.zeros((2, 3), dtype=torch.long),
    ):
        ctx = SamplingContext(
            channel_index=torch.zeros((2, 1), dtype=torch.long),
            mask=torch.ones((2, 1), dtype=torch.bool),
            repetition_index=rep,
        )
        rep_out = enc._resolve_latent_repetition_indices(sampling_ctx=ctx, batch_size=2, loc=loc)
        assert rep_out.shape == (2, enc.latent_dim)


def test_convpc_builder_direct_error_paths():
    enc = ConvPcJointEncoder(
        input_height=4,
        input_width=4,
        input_channels=1,
        latent_dim=4,
        channels=4,
        depth=2,
        kernel_size=2,
        num_repetitions=1,
        use_sum_conv=False,
        latent_depth=0,
        architecture="reference",
    )

    with pytest.raises(InvalidParameterError, match="LeafModule instances"):
        enc._validate_leaf_scope(leaf=object(), expected_scope=[0], role="x")
    with pytest.raises(InvalidParameterError, match="expected scope"):
        wrong_leaf = _default_normal_leaf([999], 2, 1)
        enc._validate_leaf_scope(leaf=wrong_leaf, expected_scope=[0], role="x")

    x_leaf = _default_normal_leaf(enc._x_cols, 2, 1)
    z_leaf = _default_normal_leaf(enc._z_cols, 2, 1)
    with pytest.raises(RuntimeError, match="Failed to infer latent injection feature width"):
        enc._build_joint_convpc_reference(
            x_leaf=x_leaf,
            z_leaf=z_leaf,
            channels=2,
            depth=2,
            kernel_size=2,
            num_repetitions=1,
            use_sum_conv=False,
            latent_depth=99,
            latent_channels=2,
            perm_latents=False,
        )

    with pytest.raises(RuntimeError, match="Latent branch was not injected"):
        enc._build_joint_convpc_legacy(
            x_leaf=x_leaf,
            z_leaf=z_leaf,
            channels=2,
            depth=2,
            kernel_size=2,
            num_repetitions=1,
            use_sum_conv=False,
            latent_depth=99,
        )


def test_convpc_nonnormal_fallback_decode_mpe_and_latent_stats_path(monkeypatch):
    enc = ConvPcJointEncoder(
        input_height=4,
        input_width=4,
        input_channels=1,
        latent_dim=4,
        channels=4,
        depth=2,
        kernel_size=2,
        num_repetitions=1,
        use_sum_conv=False,
        latent_depth=0,
        architecture="reference",
    )

    x = torch.randn(3, 1, 4, 4)
    z = torch.randn(3, 4)

    enc._z_leaf = nn.Identity()

    # Cover non-Normal encode fallback branch.
    stats, z_out = enc.encode(x, return_latent_stats=True)
    assert stats.mu.shape == (3, 4)
    assert stats.logvar.shape == (3, 4)
    assert z_out.shape == (3, 4)

    # Cover latent_stats() fallback branch.
    stats2 = enc.latent_stats(x)
    assert stats2.mu.shape == (3, 4)
    assert stats2.logvar.shape == (3, 4)

    # Cover decode(mpe=True) and sample_prior_z invalid input.
    x_dec = enc.decode(z, mpe=True)
    assert x_dec.shape == (3, 1, 4, 4)
    with pytest.raises(InvalidParameterError, match="num_samples must be >= 1"):
        enc.sample_prior_z(0)

    # Direct MC fallback error branch: posterior sampler returns wrong type.
    def _bad_posterior(*args, **kwargs):
        del args, kwargs
        return "not-a-tensor"

    monkeypatch.setattr(enc, "_posterior_sample", _bad_posterior)
    with pytest.raises(RuntimeError, match="Unexpected posterior sample type"):
        enc._latent_stats_mc_fallback(
            x_flat=torch.randn(2, 16),
            first_sample=torch.randn(2, 4),
            mpe=False,
            tau=1.0,
        )


def test_convpc_latent_stats_from_leaf_params_error_guards():
    enc = ConvPcJointEncoder(
        input_height=4,
        input_width=4,
        input_channels=1,
        latent_dim=4,
        channels=4,
        depth=2,
        kernel_size=2,
        num_repetitions=1,
        use_sum_conv=False,
        latent_depth=0,
        architecture="reference",
    )

    ctx = SamplingContext(num_samples=2)

    enc._z_leaf = nn.Identity()
    with pytest.raises(InvalidParameterError, match="require a Normal latent leaf"):
        enc._latent_stats_from_leaf_params(ctx, batch_size=2)

    # Force invalid Normal parameter dimensionality.
    bad_normal = _default_normal_leaf(enc._z_cols, out_channels=2, num_repetitions=1)
    bad_normal.loc = nn.Parameter(bad_normal.loc.unsqueeze(0))
    enc._z_leaf = bad_normal
    with pytest.raises(InvalidParameterError, match="Unexpected latent leaf parameter shape"):
        enc._latent_stats_from_leaf_params(ctx, batch_size=2)
