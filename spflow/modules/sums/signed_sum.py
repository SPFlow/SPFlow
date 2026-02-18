from __future__ import annotations

from typing import cast

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from spflow.exceptions import ShapeError, UnsupportedOperationError
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.ops.cat import Cat
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import SamplingContext, update_channel_index_strict
from spflow.utils.signed_semiring import signed_logsumexp, sign_of


class SignedSum(Module):
    """Linear-combination node that allows negative, non-normalized weights.

    This node is *not* a probabilistic mixture node. It represents a real-valued
    linear combination of input channels:

        y = Σ_j w_j * x_j

    where weights may be negative and do not need to sum to one.

    Notes:
        - `SignedSum` does not implement `log_likelihood()` because its output
          may be negative (log is undefined). Use SOCS or signed evaluation
          utilities for inference.
        - `sample()` is only supported when all weights are non-negative and
          no evidence is present, in which case it behaves like an unnormalized
          mixture over inputs.
    """

    def __init__(
        self,
        inputs: Module | list[Module],
        out_channels: int = 1,
        num_repetitions: int = 1,
        weights: Tensor | None = None,
    ) -> None:
        super().__init__()

        if not inputs:
            raise ValueError("'SignedSum' requires at least one input.")

        if isinstance(inputs, list):
            if len(inputs) == 1:
                self.inputs = inputs[0]
            else:
                self.inputs = Cat(inputs=inputs, dim=2)
        else:
            self.inputs = inputs

        self.sum_dim = 1  # sum over input channels
        self.scope = self.inputs.scope

        self.in_shape = self.inputs.out_shape
        self.out_shape = ModuleShape(
            features=self.in_shape.features, channels=out_channels, repetitions=num_repetitions
        )

        self.weights_shape = (
            self.in_shape.features,
            self.in_shape.channels,
            self.out_shape.channels,
            self.out_shape.repetitions,
        )

        if weights is None:
            weights = torch.randn(self.weights_shape, dtype=torch.get_default_dtype())
        else:
            weights = torch.as_tensor(weights, dtype=torch.get_default_dtype())

        if tuple(weights.shape) != tuple(self.weights_shape):
            raise ShapeError(
                f"Invalid shape for weights: was {tuple(weights.shape)} but expected {self.weights_shape}."
            )

        self.weights = nn.Parameter(weights)

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self.inputs.feature_to_scope

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, weights={self.weights_shape}"

    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:  # type: ignore[override]
        raise UnsupportedOperationError(
            "SignedSum does not define log_likelihood() because outputs may be negative. "
            "Use SOCS (sum of squares) or signed evaluation utilities."
        )

    def _expectation_maximization_step(
        self,
        data: Tensor,
        bias_correction: bool = True,
        *,
        cache: Cache,
    ) -> None:
        raise UnsupportedOperationError("SignedSum does not support expectation-maximization.")

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        raise UnsupportedOperationError("SignedSum.marginalize() is not supported.")

    @cached
    def signed_logabs_and_sign(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate this node in ``(log|·|, sign)`` form.

        Returns:
            logabs: Tensor of shape (B, F, OC, R)
            sign: Tensor of shape (B, F, OC, R) in {-1,0,+1}
        """
        # Cache keying: re-use `Cache` internal dicts with a custom method name.
        cached_ = cache.get("signed_logabs_and_sign", self)
        if cached_ is not None:
            return cached_

        # Evaluate children in signed semiring
        child = self.inputs
        if hasattr(child, "signed_logabs_and_sign"):
            child_logabs, child_sign = child.signed_logabs_and_sign(data, cache=cache)  # type: ignore[attr-defined]
        else:
            # Recursively evaluate mixed sub-graphs (e.g., Product/Cat containing SignedSum leaves).
            from spflow.modules.sos.socs import _signed_eval

            child_logabs, child_sign = _signed_eval(cast(Module, child), data, cache)

        # child_* shapes: (B, F, IC, R)
        if child_logabs.dim() != 4:
            raise ShapeError(
                f"Expected child signed evaluation to be 4D (B,F,C,R), got shape {tuple(child_logabs.shape)}."
            )

        ll = rearrange(child_logabs, "b f ci r -> b f ci 1 r")
        ss = rearrange(child_sign, "b f ci r -> b f ci 1 r").to(dtype=torch.int8)

        w = rearrange(self.weights, "f ci co r -> 1 f ci co r")
        w_sign = sign_of(w).to(dtype=torch.int8)
        # Avoid log(0) for zero weights; zero-weight terms should contribute nothing.
        w_logabs = torch.log(torch.abs(w).clamp_min(1e-30))

        # Terms: w * x = sign(w)*sign(x) * exp(log|w|+log|x|)
        term_sign = (w_sign * ss).to(dtype=torch.int8)
        term_logabs = ll + w_logabs  # broadcast over batch

        out_logabs, out_sign = signed_logsumexp(
            logabs_terms=term_logabs,
            sign_terms=term_sign,
            dim=self.sum_dim + 1,  # reduce IC
            keepdim=False,
            eps=0.0,
        )

        cache.set("signed_logabs_and_sign", self, (out_logabs, out_sign))
        return out_logabs, out_sign

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
    ) -> Tensor:
        """Sample from the unnormalized non-negative subset of SignedSum.

        Only supported if all weights are >= 0 and no evidence is provided.
        """
        if torch.isfinite(data).any():
            raise UnsupportedOperationError(
                "SignedSum.sample() does not support conditional sampling with evidence."
            )

        if (self.weights < 0).any():
            raise UnsupportedOperationError(
                "SignedSum.sample() is only supported when all weights are non-negative."
            )

        # Only supports scalar feature routing like Sum: choose input-channel per feature.
        # We treat weights as proportional probabilities.
        w = self.weights
        if w.dim() != 4:
            raise ShapeError(f"Expected weights to be 4D, got shape {tuple(w.shape)}.")
        if self.out_shape.repetitions != 1:
            raise UnsupportedOperationError(
                "SignedSum.sample() currently supports num_repetitions == 1 only."
            )

        # Current output channel selection from parent
        batch_size = int(data.shape[0])
        num_weight_features = int(w.shape[0])
        in_channels_total = w.shape[1]
        context_features = int(sampling_ctx.channel_index.shape[1])
        if context_features == num_weight_features:
            oidx = repeat(sampling_ctx.channel_index, "b f -> b f ci 1", ci=in_channels_total)
        elif context_features == 1:
            oidx = repeat(sampling_ctx.channel_index, "b 1 -> b f ci 1", f=num_weight_features, ci=in_channels_total)
        else:
            raise ShapeError(
                f"Expected channel_index feature width 1 or {num_weight_features}, got {context_features}."
            )
        w_sel = repeat(w[..., 0], "f ci co -> b f ci co", b=batch_size)
        w_sel = w_sel.gather(dim=3, index=oidx)
        w_sel = rearrange(w_sel, "b f ci 1 -> b f ci")

        # Normalize over input channels
        probs = w_sel / w_sel.sum(dim=2, keepdim=True).clamp_min(1e-12)
        if sampling_ctx.is_mpe:
            new_channel_index = torch.argmax(probs, dim=-1)
        else:
            new_channel_index = torch.distributions.Categorical(probs=probs).sample()

        update_channel_index_strict(sampling_ctx, new_channel_index)
        self.inputs._sample(data=data, cache=cache, sampling_ctx=sampling_ctx)
        return data
