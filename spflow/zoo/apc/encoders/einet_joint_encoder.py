"""Einet-based APC joint encoder over data and latent variables.

The encoder builds a joint PC over concatenated variables ``[X, Z]`` using two
leaf modules:
- one leaf that covers all observed ``X`` columns,
- one leaf that covers all latent ``Z`` columns.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterError, ShapeError, UnsupportedOperationError
from spflow.modules.leaves import Normal
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from spflow.zoo.apc.encoders.base import LatentStats
from spflow.zoo.einet import Einet

LeafFactory = Callable[[list[int], int, int], LeafModule]


def _default_normal_leaf(scope_indices: list[int], out_channels: int, num_repetitions: int) -> LeafModule:
    """Create a Normal leaf with reference-style ``(mu, logvar)`` init."""
    event_shape = (len(scope_indices), out_channels, num_repetitions)
    loc = torch.randn(event_shape)
    logvar = torch.randn(event_shape)
    scale = torch.exp(0.5 * logvar)
    return Normal(
        scope=scope_indices,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
        loc=loc,
        scale=scale,
    )


class _LatentSelectionCapture(Module):
    """Identity wrapper that records latent leaf routing indices."""

    def __init__(
        self,
        inputs: Module,
        capture_fn: Callable[[Tensor, Tensor | None], None],
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.scope = inputs.scope
        self.in_shape = inputs.out_shape
        self.out_shape = inputs.out_shape
        self._capture_fn = capture_fn

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self.inputs.feature_to_scope

    @property
    def loc(self) -> Tensor:
        """Forward latent-mean parameter access for test parity helpers."""
        return self.inputs.loc  # type: ignore[return-value]

    @property
    def log_scale(self) -> Tensor:
        """Forward latent-log-scale parameter access for test parity helpers."""
        return self.inputs.log_scale  # type: ignore[return-value]

    def _align_feature_width(
        self,
        tensor: Tensor,
        *,
        target_features: int,
        data_num_features: int,
        scope_cols: list[int],
        kind: str,
    ) -> Tensor:
        if tensor.dim() not in (2, 3):
            raise ShapeError(
                f"Latent selection capture {kind} tensor must have rank 2 or 3, got rank {tensor.dim()}."
            )
        if tensor.shape[1] == 1:
            if tensor.dim() == 3:
                return repeat(tensor, "b 1 c -> b f c", f=target_features)
            return repeat(tensor, "b 1 -> b f", f=target_features)
        if tensor.shape[1] == target_features:
            return tensor
        if tensor.shape[1] == data_num_features:
            if tensor.dim() == 3:
                return tensor[:, scope_cols, :]
            return tensor[:, scope_cols]
        raise ShapeError(
            f"Latent selection capture {kind} width mismatch: "
            f"got {tensor.shape[1]}, expected 1, {target_features}, or {data_num_features}."
        )

    def _resolve_channel_index(
        self,
        channel_index: Tensor,
        *,
        target_features: int,
        data_num_features: int,
        scope_cols: list[int],
    ) -> Tensor:
        return self._align_feature_width(
            channel_index,
            target_features=target_features,
            data_num_features=data_num_features,
            scope_cols=scope_cols,
            kind="channel",
        )

    def _resolve_repetition_index(
        self,
        rep_idx: Tensor | None,
        *,
        target_features: int,
        data_num_features: int,
        scope_cols: list[int],
    ) -> Tensor | None:
        if rep_idx is None:
            return None
        if (
            rep_idx.dim() == 2
            and rep_idx.is_floating_point()
            and rep_idx.shape[1] == self.out_shape.repetitions
        ):
            return rep_idx
        if rep_idx.dim() == 1:
            return repeat(rep_idx, "b -> b f", f=target_features)
        if rep_idx.dim() != 2:
            raise ShapeError(
                "Latent selection capture repetition index must have rank 1 or 2, "
                f"got rank {rep_idx.dim()}."
            )
        return self._align_feature_width(
            rep_idx,
            target_features=target_features,
            data_num_features=data_num_features,
            scope_cols=scope_cols,
            kind="repetition",
        )

    def _capture(self, sampling_ctx: SamplingContext, data_num_features: int) -> None:
        target_features = self.in_shape.features
        scope_cols = list(self.scope.query)
        channel_idx = self._resolve_channel_index(
            sampling_ctx.channel_index,
            target_features=target_features,
            data_num_features=data_num_features,
            scope_cols=scope_cols,
        )
        repetition_index = self._resolve_repetition_index(
            sampling_ctx.repetition_index,
            target_features=target_features,
            data_num_features=data_num_features,
            scope_cols=scope_cols,
        )
        self._capture_fn(
            channel_idx.clone(),
            None if repetition_index is None else repetition_index.clone(),
        )

    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:
        return self.inputs.log_likelihood(data, cache=cache)

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
    ) -> Tensor:
        sampling_ctx.validate_sampling_context(
            num_samples=data.shape[0],
            num_features=self.out_shape.features,
            num_channels=self.out_shape.channels,
            num_repetitions=self.out_shape.repetitions,
            allowed_feature_widths=(1, self.out_shape.features, data.shape[1]),
        )
        self._capture(sampling_ctx, data.shape[1])
        return self.inputs._sample(data=data, cache=cache, sampling_ctx=sampling_ctx)

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        del marg_rvs, prune, cache
        raise NotImplementedError("Marginalization is not implemented for _LatentSelectionCapture.")


class EinetJointEncoder(nn.Module):
    """Joint APC encoder using an Einet over concatenated features ``[x, z]``."""

    def __init__(
        self,
        *,
        num_x_features: int,
        latent_dim: int,
        num_sums: int = 10,
        num_leaves: int = 10,
        depth: int = 1,
        num_repetitions: int = 5,
        layer_type: Literal["einsum", "linsum"] = "linsum",
        structure: Literal["top-down", "bottom-up"] = "top-down",
        x_leaf_factory: LeafFactory | None = None,
        z_leaf_factory: LeafFactory | None = None,
        posterior_stat_samples: int = 4,
        posterior_var_floor: float = 1e-6,
    ) -> None:
        """Initialize a joint Einet-based APC encoder.

        Args:
            num_x_features: Number of flattened data features in ``X``.
            latent_dim: Number of latent dimensions in ``Z``.
            num_sums: Number of sum units per sum layer.
            num_leaves: Number of leaf channels/components.
            depth: Number of internal Einet product/sum stages.
            num_repetitions: Number of repetitions/channels in the circuit.
            layer_type: Internal sum-product implementation type.
            structure: Einet structure type. Must be ``"top-down"`` for APC sampling.
            x_leaf_factory: Factory to create the ``X`` leaf module.
            z_leaf_factory: Factory to create the ``Z`` leaf module.
            posterior_stat_samples: Number of posterior samples used to estimate moments.
            posterior_var_floor: Numerical floor for latent variance estimates.
        """
        super().__init__()

        if num_x_features <= 0:
            raise InvalidParameterError(f"num_x_features must be >= 1, got {num_x_features}.")
        if latent_dim <= 0:
            raise InvalidParameterError(f"latent_dim must be >= 1, got {latent_dim}.")
        if posterior_stat_samples <= 0:
            raise InvalidParameterError(f"posterior_stat_samples must be >= 1, got {posterior_stat_samples}.")
        if posterior_var_floor <= 0.0:
            raise InvalidParameterError(f"posterior_var_floor must be > 0, got {posterior_var_floor}.")
        if structure != "top-down":
            raise InvalidParameterError(
                "EinetJointEncoder requires structure='top-down' because APC encoding/decoding uses sampling."
            )

        self.num_x_features = num_x_features
        self.latent_dim = latent_dim
        self.posterior_stat_samples = posterior_stat_samples
        self.posterior_var_floor = posterior_var_floor

        self._x_cols = list(range(num_x_features))
        self._z_cols = list(range(num_x_features, num_x_features + latent_dim))

        x_leaf_factory = x_leaf_factory or _default_normal_leaf
        z_leaf_factory = z_leaf_factory or _default_normal_leaf

        x_leaf = x_leaf_factory(self._x_cols, num_leaves, num_repetitions)
        self._validate_leaf_scope(leaf=x_leaf, expected_scope=self._x_cols, role="x")

        z_leaf = z_leaf_factory(self._z_cols, num_leaves, num_repetitions)
        self._validate_leaf_scope(leaf=z_leaf, expected_scope=self._z_cols, role="z")
        self._z_leaf = z_leaf
        self._last_latent_leaf_channel_index: Tensor | None = None
        self._last_latent_leaf_repetition_index: Tensor | None = None

        z_leaf_capture = _LatentSelectionCapture(
            inputs=z_leaf,
            capture_fn=self._record_latent_leaf_selection,
        )

        self.pc = Einet(
            leaf_modules=[x_leaf, z_leaf_capture],
            num_classes=1,
            num_sums=num_sums,
            num_leaves=num_leaves,
            depth=depth,
            num_repetitions=num_repetitions,
            layer_type=layer_type,
            structure=structure,
        )

    @staticmethod
    def _validate_leaf_scope(*, leaf: LeafModule, expected_scope: list[int], role: str) -> None:
        """Validate that a leaf covers exactly the expected variable scope."""
        if not isinstance(leaf, LeafModule):
            raise InvalidParameterError(
                f"{role}_leaf_factory must return LeafModule instances, got {type(leaf)}."
            )
        scope_query = list(leaf.scope.query)
        if set(scope_query) != set(expected_scope) or len(scope_query) != len(expected_scope):
            raise InvalidParameterError(
                f"{role}_leaf_factory returned scope {scope_query}, expected scope {expected_scope}."
            )

    def _record_latent_leaf_selection(
        self,
        channel_index: Tensor,
        repetition_index: Tensor | None,
    ) -> None:
        """Capture leaf-level latent routing signals from the sampling path."""
        self._last_latent_leaf_channel_index = channel_index
        self._last_latent_leaf_repetition_index = repetition_index

    def _reset_latent_leaf_selection(self) -> None:
        self._last_latent_leaf_channel_index = None
        self._last_latent_leaf_repetition_index = None

    def _flatten_x(self, x: Tensor) -> Tensor:
        """Flatten ``x`` to ``(B, num_x_features)`` and validate dimensionality."""
        if x.dim() < 2:
            raise ShapeError(f"x must have at least 2 dimensions, got shape {tuple(x.shape)}.")
        x_flat = rearrange(x, "b ... -> b (...)")
        if x_flat.shape[1] != self.num_x_features:
            raise ShapeError(
                f"Expected x to have {self.num_x_features} flattened features, got {x_flat.shape[1]}."
            )
        return x_flat

    def _flatten_z(self, z: Tensor) -> Tensor:
        """Flatten ``z`` to ``(B, latent_dim)`` and validate dimensionality."""
        if z.dim() < 2:
            raise ShapeError(f"z must have at least 2 dimensions, got shape {tuple(z.shape)}.")
        z_flat = rearrange(z, "b ... -> b (...)")
        if z_flat.shape[1] != self.latent_dim:
            raise ShapeError(f"Expected z to have latent_dim={self.latent_dim}, got {z_flat.shape[1]}.")
        return z_flat

    @staticmethod
    def _evidence_dtype(*, x_flat: Tensor | None, z_flat: Tensor | None) -> torch.dtype:
        """Select an evidence dtype from provided tensors, falling back to default dtype."""
        if x_flat is not None and x_flat.is_floating_point():
            return x_flat.dtype
        if z_flat is not None and z_flat.is_floating_point():
            return z_flat.dtype
        return torch.get_default_dtype()

    def _build_evidence(
        self,
        *,
        x_flat: Tensor | None,
        z_flat: Tensor | None,
        num_samples: int | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        """Build a joint evidence tensor over ``[X, Z]`` with ``NaN`` for missing blocks.

        ``None`` inputs are converted to all-``NaN`` blocks, enabling conditional
        likelihood/sampling in the underlying PC.
        """
        if x_flat is None and z_flat is None and num_samples is None:
            raise InvalidParameterError("num_samples must be provided when x_flat and z_flat are None.")

        inferred_batch = num_samples
        if x_flat is not None:
            inferred_batch = x_flat.shape[0]
        if z_flat is not None:
            if inferred_batch is None:
                inferred_batch = z_flat.shape[0]
            elif z_flat.shape[0] != inferred_batch:
                raise ShapeError(
                    f"x and z batch sizes must match, got {inferred_batch} and {z_flat.shape[0]}."
                )

        if inferred_batch is None:
            raise RuntimeError("Failed to infer batch size for evidence construction.")

        if device is None:
            if x_flat is not None:
                device = x_flat.device
            elif z_flat is not None:
                device = z_flat.device
            else:
                device = self.pc.device

        dtype = self._evidence_dtype(x_flat=x_flat, z_flat=z_flat)

        if x_flat is None:
            x_flat = torch.full((inferred_batch, self.num_x_features), torch.nan, device=device, dtype=dtype)
        else:
            x_flat = x_flat.to(device=device, dtype=dtype)

        if z_flat is None:
            z_flat = torch.full((inferred_batch, self.latent_dim), torch.nan, device=device, dtype=dtype)
        else:
            z_flat = z_flat.to(device=device, dtype=dtype)

        return torch.cat([x_flat, z_flat], dim=1)

    @staticmethod
    def _flatten_ll(ll: Tensor) -> Tensor:
        """Normalize PC log-likelihood outputs to shape ``(B,)``."""
        if ll.dim() < 1:
            raise ShapeError(f"Expected log-likelihood with batch dimension, got shape {tuple(ll.shape)}.")
        ll_flat = rearrange(ll, "b ... -> b (...)")
        if ll_flat.shape[1] != 1:
            raise ShapeError(
                f"Expected scalar log-likelihood per sample, got trailing shape {tuple(ll_flat.shape[1:])}."
            )
        return ll_flat[:, 0]

    def _posterior_sample(
        self, x_flat: Tensor, *, mpe: bool, tau: float, return_sampling_ctx: bool = False
    ) -> Tensor | tuple[Tensor, SamplingContext]:
        """Sample ``z ~ p(Z|X=x)`` and optionally return sampling context."""
        self._reset_latent_leaf_selection()
        evidence = self._build_evidence(x_flat=x_flat, z_flat=None)
        joint, sampling_ctx = self._sample_joint(
            evidence=evidence,
            mpe=mpe,
            tau=tau,
            return_sampling_ctx=return_sampling_ctx,
        )
        z = joint[:, self._z_cols]
        if return_sampling_ctx:
            assert sampling_ctx is not None
            return z, sampling_ctx
        return z

    def _sample_joint(
        self,
        *,
        evidence: Tensor,
        mpe: bool,
        tau: float,
        return_sampling_ctx: bool,
    ) -> tuple[Tensor, SamplingContext | None]:
        """Sample the joint Einet with differentiable routing semantics."""
        cache = Cache()
        # Populate evidence-conditioned likelihood cache before top-down sampling.
        self.pc.log_likelihood(evidence, cache=cache)
        batch_size = evidence.shape[0]
        channel_index = torch.ones((batch_size, 1, 1), dtype=torch.get_default_dtype(), device=evidence.device)
        mask = torch.full((batch_size, 1), True, dtype=torch.bool, device=evidence.device)
        repetition_index = torch.zeros(
            (batch_size, self.pc.num_repetitions),
            dtype=torch.get_default_dtype(),
            device=evidence.device,
        )
        repetition_index[:, 0] = 1.0
        sampling_ctx: SamplingContext | None = SamplingContext(
            channel_index=channel_index,
            mask=mask,
            device=evidence.device,
            repetition_index=repetition_index,
            is_mpe=mpe,
            is_differentiable=True,
            tau=tau,
        )
        joint = self.pc._sample(data=evidence, cache=cache, sampling_ctx=sampling_ctx)
        if return_sampling_ctx:
            return joint, sampling_ctx
        return joint, None

    @staticmethod
    def _normalize_probs(probs: Tensor, *, dim: int) -> Tensor:
        """Normalize non-negative probabilities along ``dim``."""
        eps = torch.finfo(probs.dtype).eps
        return probs / probs.sum(dim=dim, keepdim=True).clamp_min(eps)

    def _align_latent_feature_width(
        self,
        tensor: Tensor,
        *,
        target_features: int,
        total_features: int,
    ) -> Tensor:
        """Align feature axis to latent width for routing tensors."""
        if tensor.shape[1] == target_features:
            return tensor
        if tensor.shape[1] == 1:
            if tensor.dim() == 3:
                return repeat(tensor, "b 1 c -> b f c", f=target_features)
            return repeat(tensor, "b 1 -> b f", f=target_features)
        if tensor.shape[1] == total_features:
            if tensor.dim() == 3:
                return tensor[:, self._z_cols, :]
            return tensor[:, self._z_cols]
        raise ShapeError(
            "Latent routing feature mismatch: "
            f"got {tensor.shape[1]}, expected 1, {target_features}, or {total_features}."
        )

    def _resolve_latent_channel_weights(
        self,
        *,
        sampling_ctx: SamplingContext,
        batch_size: int,
        loc: Tensor,
    ) -> Tensor:
        """Resolve latent channel routing as probabilities of shape ``(B, F, C)``."""
        channels = self._last_latent_leaf_channel_index
        if channels is None:
            channels = sampling_ctx.channel_index
        if channels.shape[0] != batch_size:
            raise ShapeError(
                f"Latent channel routing batch mismatch: expected {batch_size}, got {channels.shape[0]}."
            )
        channels = self._align_latent_feature_width(
            channels,
            target_features=self.latent_dim,
            total_features=self.num_x_features + self.latent_dim,
        )
        if channels.dim() == 2:
            if channels.dtype == torch.bool or channels.is_floating_point() or channels.is_complex():
                raise InvalidParameterError(
                    "Integer latent channel indices are required for rank-2 channel routing."
                )
            channels_long = channels.to(device=loc.device, dtype=torch.long).clamp(
                min=0, max=loc.shape[1] - 1
            )
            return torch.nn.functional.one_hot(channels_long, num_classes=loc.shape[1]).to(dtype=loc.dtype)
        if channels.dim() != 3:
            raise ShapeError(f"Latent channel routing must have rank 2 or 3, got rank {channels.dim()}.")
        if channels.shape[2] == 1 and loc.shape[1] > 1:
            expanded = channels.new_zeros((channels.shape[0], channels.shape[1], loc.shape[1]))
            expanded[:, :, 0] = channels[:, :, 0]
            channels = expanded
        elif channels.shape[2] != loc.shape[1]:
            raise ShapeError(
                "Latent channel routing channel-axis mismatch: "
                f"got {channels.shape[2]}, expected {loc.shape[1]}."
            )
        return self._normalize_probs(channels.to(device=loc.device, dtype=loc.dtype), dim=-1)

    def _resolve_latent_repetition_weights(
        self,
        *,
        sampling_ctx: SamplingContext,
        batch_size: int,
        loc: Tensor,
    ) -> Tensor:
        """Resolve latent repetition routing as probabilities of shape ``(B, F, R)``."""
        rep = self._last_latent_leaf_repetition_index
        if rep is None:
            rep = sampling_ctx.repetition_index
        if rep is None:
            base = torch.zeros((batch_size,), dtype=torch.long, device=loc.device)
            rep_probs = torch.nn.functional.one_hot(base, num_classes=loc.shape[2]).to(dtype=loc.dtype)
            return repeat(rep_probs, "b r -> b f r", f=self.latent_dim)
        if rep.shape[0] != batch_size:
            raise ShapeError(
                f"Latent repetition routing batch mismatch: expected {batch_size}, got {rep.shape[0]}."
            )
        if rep.dim() == 1:
            rep_long = rep.to(device=loc.device, dtype=torch.long).clamp(min=0, max=loc.shape[2] - 1)
            rep_probs = torch.nn.functional.one_hot(rep_long, num_classes=loc.shape[2]).to(dtype=loc.dtype)
            return repeat(rep_probs, "b r -> b f r", f=self.latent_dim)
        if rep.dim() != 2:
            raise ShapeError(f"Latent repetition routing must have rank 1 or 2, got rank {rep.dim()}.")
        if rep.is_floating_point() and rep.shape[1] == loc.shape[2]:
            rep_probs = self._normalize_probs(rep.to(device=loc.device, dtype=loc.dtype), dim=-1)
            return repeat(rep_probs, "b r -> b f r", f=self.latent_dim)

        rep_aligned = self._align_latent_feature_width(
            rep,
            target_features=self.latent_dim,
            total_features=self.num_x_features + self.latent_dim,
        )
        if rep_aligned.dim() != 2:
            raise ShapeError(
                "Latent repetition feature routing must be rank-2 after alignment, "
                f"got rank {rep_aligned.dim()}."
            )
        rep_long = rep_aligned.to(device=loc.device, dtype=torch.long).clamp(min=0, max=loc.shape[2] - 1)
        return torch.nn.functional.one_hot(rep_long, num_classes=loc.shape[2]).to(dtype=loc.dtype)

    def _latent_stats_from_leaf_params(self, sampling_ctx: SamplingContext, batch_size: int) -> LatentStats:
        """Extract posterior ``mu/logvar`` from selected latent leaf parameters."""
        if not isinstance(self._z_leaf, Normal):
            raise UnsupportedOperationError(
                "Latent leaf is not Normal; falling back to Monte Carlo latent stats estimation."
            )

        loc = self._z_leaf.loc
        log_scale = self._z_leaf.log_scale
        if loc.dim() != 3 or log_scale.dim() != 3:
            raise ShapeError(
                "Expected Normal latent leaf params to have shape (features, channels, repetitions), "
                f"got loc {tuple(loc.shape)} and log_scale {tuple(log_scale.shape)}."
            )
        if loc.shape != log_scale.shape:
            raise ShapeError(
                f"Normal latent leaf params shape mismatch: loc {tuple(loc.shape)} vs log_scale {tuple(log_scale.shape)}."
            )
        if loc.shape[0] != self.latent_dim:
            raise ShapeError(
                f"Normal latent leaf feature mismatch: expected latent_dim={self.latent_dim}, got {loc.shape[0]}."
            )

        channel_weights = self._resolve_latent_channel_weights(
            sampling_ctx=sampling_ctx,
            batch_size=batch_size,
            loc=loc,
        )
        repetition_weights = self._resolve_latent_repetition_weights(
            sampling_ctx=sampling_ctx,
            batch_size=batch_size,
            loc=loc,
        )
        mu = torch.einsum("bfc,bfr,fcr->bf", channel_weights, repetition_weights, loc)
        logvar = 2.0 * torch.einsum("bfc,bfr,fcr->bf", channel_weights, repetition_weights, log_scale)

        if mu.shape != (batch_size, self.latent_dim):
            raise ShapeError(
                f"Unexpected latent mean shape {tuple(mu.shape)}; expected ({batch_size}, {self.latent_dim})."
            )
        if logvar.shape != mu.shape:
            raise ShapeError(
                f"Unexpected latent logvar shape {tuple(logvar.shape)}; expected {tuple(mu.shape)}."
            )
        return LatentStats(mu=mu, logvar=logvar)

    def _latent_stats_mc_fallback(
        self,
        *,
        x_flat: Tensor,
        first_sample: Tensor,
        mpe: bool,
        tau: float,
    ) -> LatentStats:
        """Fallback posterior stats estimation for non-Normal latent leaves."""
        samples = [first_sample]
        for _ in range(self.posterior_stat_samples - 1):
            sample_i = self._posterior_sample(x_flat, mpe=mpe, tau=tau, return_sampling_ctx=False)
            if not isinstance(sample_i, Tensor):
                raise RuntimeError("Unexpected posterior sample type in MC stats fallback.")
            samples.append(sample_i)

        z_stack = torch.stack(samples, dim=0)
        mu = z_stack.mean(dim=0)
        var = z_stack.var(dim=0, unbiased=False).clamp_min(self.posterior_var_floor)
        return LatentStats(mu=mu, logvar=var.log())

    def encode(
        self,
        x: Tensor,
        *,
        mpe: bool = False,
        tau: float = 1.0,
        return_latent_stats: bool = False,
    ) -> Tensor | tuple[LatentStats, Tensor]:
        """Encode observations into latent samples.

        Args:
            x: Observation tensor.
            mpe: Whether to use deterministic MPE routing. When ``True``, returned
                ``z`` is deterministic even if ``return_latent_stats=True``.
            tau: Temperature for differentiable sampling. Used for stochastic
                sampling when ``mpe=False``.
            return_latent_stats: When ``True``, return ``(LatentStats, z)``.

        Returns:
            Either ``z`` or ``(LatentStats, z)``.
        """
        x_flat = self._flatten_x(x)
        if not return_latent_stats:
            z = self._posterior_sample(x_flat, mpe=mpe, tau=tau, return_sampling_ctx=False)
            return z

        z_and_ctx = self._posterior_sample(x_flat, mpe=mpe, tau=tau, return_sampling_ctx=True)
        if not isinstance(z_and_ctx, tuple):
            raise RuntimeError("Expected (z, sampling_ctx) from posterior sampling.")
        z_first, sampling_ctx = z_and_ctx
        batch_size = z_first.shape[0]
        try:
            stats = self._latent_stats_from_leaf_params(sampling_ctx=sampling_ctx, batch_size=batch_size)
        except UnsupportedOperationError:
            stats = self._latent_stats_mc_fallback(
                x_flat=x_flat,
                first_sample=z_first,
                mpe=mpe,
                tau=tau,
            )
        if mpe:
            z_samples = z_first
        else:
            z_samples = tau * torch.randn_like(stats.mu) * torch.exp(0.5 * stats.logvar) + stats.mu
        return stats, z_samples

    def decode(
        self,
        z: Tensor,
        *,
        x: Tensor | None = None,
        mpe: bool = False,
        tau: float = 1.0,
        fill_evidence: bool = False,
    ) -> Tensor:
        """Decode latents by sampling/imputing the ``X`` block given ``Z`` evidence.

        Args:
            z: Latent tensor.
            x: Optional partial data evidence. Finite values are observed entries.
            mpe: Whether to use deterministic MPE routing.
            tau: Temperature for differentiable sampling.
            fill_evidence: If ``True``, preserve finite entries from ``x`` in output.
        """
        z_flat = self._flatten_z(z)
        x_flat = None if x is None else self._flatten_x(x)
        evidence = self._build_evidence(x_flat=x_flat, z_flat=z_flat)
        joint, _ = self._sample_joint(evidence=evidence, mpe=mpe, tau=tau, return_sampling_ctx=False)

        x_rec = joint[:, self._x_cols]
        if fill_evidence and x_flat is not None:
            finite_mask = torch.isfinite(x_flat)
            x_rec = torch.where(finite_mask, x_flat.to(x_rec.dtype), x_rec)
        return x_rec

    def joint_log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        """Compute per-sample joint log-likelihood ``log p(x, z)``."""
        x_flat = self._flatten_x(x)
        z_flat = self._flatten_z(z)
        evidence = self._build_evidence(x_flat=x_flat, z_flat=z_flat)
        ll = self.pc.log_likelihood(evidence)
        return self._flatten_ll(ll)

    def log_likelihood_x(self, x: Tensor) -> Tensor:
        """Compute per-sample marginal log-likelihood ``log p(x)``."""
        x_flat = self._flatten_x(x)
        evidence = self._build_evidence(x_flat=x_flat, z_flat=None)
        ll = self.pc.log_likelihood(evidence)
        return self._flatten_ll(ll)

    def sample_prior_z(self, num_samples: int, *, tau: float = 1.0) -> Tensor:
        """Sample latent variables from the model prior over ``Z``."""
        if num_samples <= 0:
            raise InvalidParameterError(f"num_samples must be >= 1, got {num_samples}.")
        evidence = self._build_evidence(
            x_flat=None, z_flat=None, num_samples=num_samples, device=self.pc.device
        )
        joint, _ = self._sample_joint(evidence=evidence, mpe=False, tau=tau, return_sampling_ctx=False)
        z = joint[:, self._z_cols]
        return z

    def latent_stats(self, x: Tensor, *, tau: float = 1.0) -> LatentStats:
        """Return latent stats from selected latent leaf parameters."""
        x_flat = self._flatten_x(x)
        z_and_ctx = self._posterior_sample(x_flat, mpe=False, tau=tau, return_sampling_ctx=True)
        if not isinstance(z_and_ctx, tuple):
            raise RuntimeError("Expected (z, sampling_ctx) from posterior sampling.")
        z_first, sampling_ctx = z_and_ctx
        try:
            return self._latent_stats_from_leaf_params(sampling_ctx=sampling_ctx, batch_size=z_first.shape[0])
        except UnsupportedOperationError:
            return self._latent_stats_mc_fallback(
                x_flat=x_flat,
                first_sample=z_first,
                mpe=False,
                tau=tau,
            )
