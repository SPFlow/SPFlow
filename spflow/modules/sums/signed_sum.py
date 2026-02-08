from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from spflow.exceptions import ShapeError, UnsupportedOperationError
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.ops.cat import Cat
from spflow.utils.cache import Cache
from spflow.utils.diff_sampling import DiffSampleMethod, sample_categorical_differentiably
from spflow.utils.diff_sampling_context import DifferentiableSamplingContext
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context
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

    def expectation_maximization(
        self,
        data: Tensor,
        bias_correction: bool = True,
        cache: Cache | None = None,
    ) -> None:
        raise UnsupportedOperationError("SignedSum does not support expectation-maximization.")

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Tensor | None = None,
        cache: Cache | None = None,
    ) -> None:
        raise UnsupportedOperationError("SignedSum does not support maximum-likelihood estimation.")

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        raise UnsupportedOperationError("SignedSum.marginalize() is not supported.")

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
        if cache is None:
            cache = Cache()

        # Cache keying: re-use `Cache` internal dicts with a custom method name.
        cached = cache.get("signed_logabs_and_sign", self)
        if cached is not None:
            return cached

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

        ll = child_logabs.unsqueeze(3)  # (B, F, IC, 1, R)
        ss = child_sign.unsqueeze(3).to(dtype=torch.int8)  # (B, F, IC, 1, R)

        w = self.weights.unsqueeze(0)  # (1, F, IC, OC, R)
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

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Sample from the unnormalized non-negative subset of SignedSum.

        Only supported if all weights are >= 0 and no evidence is provided.
        """
        if cache is None:
            cache = Cache()

        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full((num_samples, len(self.scope.query)), float("nan"), device=self.device)

        if torch.isfinite(data).any():
            raise UnsupportedOperationError(
                "SignedSum.sample() does not support conditional sampling with evidence."
            )

        if (self.weights < 0).any():
            raise UnsupportedOperationError(
                "SignedSum.sample() is only supported when all weights are non-negative."
            )

        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)

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
        oidx = sampling_ctx.channel_index[..., None, None]
        in_channels_total = w.shape[1]
        oidx = oidx.expand(-1, w.shape[0], in_channels_total, -1)
        w_sel = w[..., 0].unsqueeze(0).expand(data.shape[0], -1, -1, -1).gather(dim=3, index=oidx).squeeze(3)

        # Normalize over input channels
        probs = w_sel / w_sel.sum(dim=2, keepdim=True).clamp_min(1e-12)
        if is_mpe:
            new_channel_index = torch.argmax(probs, dim=-1)
        else:
            new_channel_index = torch.distributions.Categorical(probs=probs).sample()

        # Update sampling context and delegate to child
        sampling_ctx.update(
            channel_index=new_channel_index, mask=sampling_ctx.mask.expand_as(new_channel_index)
        )
        self.inputs.sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=sampling_ctx)
        return data

    def rsample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
        method: str = "simple",
        tau: float = 1.0,
        hard: bool = True,
    ) -> Tensor:
        """Differentiable sampling for non-negative SignedSum subset."""
        if cache is None:
            cache = Cache()
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full((num_samples, len(self.scope.query)), float("nan"), device=self.device)

        if torch.isfinite(data).any():
            raise UnsupportedOperationError("SignedSum.rsample() does not support conditional evidence.")
        if (self.weights < 0).any():
            raise UnsupportedOperationError("SignedSum.rsample() only supports non-negative weights.")
        if self.out_shape.repetitions != 1:
            raise UnsupportedOperationError(
                "SignedSum.rsample() currently supports num_repetitions == 1 only."
            )

        if sampling_ctx is None or not isinstance(sampling_ctx, DifferentiableSamplingContext):
            sampling_ctx = DifferentiableSamplingContext(
                channel_index=torch.zeros(
                    (data.shape[0], self.out_shape.features), dtype=torch.long, device=data.device
                ),
                mask=torch.ones(
                    (data.shape[0], self.out_shape.features), dtype=torch.bool, device=data.device
                ),
                repetition_index=None,
                method=DiffSampleMethod(method),
                tau=tau,
                hard=hard,
            )
        else:
            sampling_ctx = sampling_ctx

        oidx = sampling_ctx.channel_index[..., None, None]
        in_channels_total = self.weights.shape[1]
        oidx = oidx.expand(-1, self.weights.shape[0], in_channels_total, -1)
        w_sel = (
            self.weights[..., 0]
            .unsqueeze(0)
            .expand(data.shape[0], -1, -1, -1)
            .gather(dim=3, index=oidx)
            .squeeze(3)
        )
        probs = w_sel / w_sel.sum(dim=2, keepdim=True).clamp_min(1e-12)
        logits = torch.log(probs.clamp_min(1e-12))

        selector = sample_categorical_differentiably(
            dim=-1,
            is_mpe=is_mpe,
            hard=hard,
            tau=tau,
            logits=logits,
            method=DiffSampleMethod(method),
        )
        new_channel_index = selector.argmax(dim=-1)
        sampling_ctx.update(
            channel_index=new_channel_index, mask=sampling_ctx.mask.expand_as(new_channel_index)
        )
        sampling_ctx.channel_select = selector
        sampling_ctx.method = DiffSampleMethod(method)
        sampling_ctx.tau = tau
        sampling_ctx.hard = hard

        return self.inputs.rsample(
            data=data,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
            method=method,
            tau=tau,
            hard=hard,
        )
