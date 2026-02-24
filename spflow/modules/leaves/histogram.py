from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterCombinationError, InvalidParameterError
from spflow.meta.data import Scope
from spflow.modules.leaves.leaf import LeafModule
from spflow.utils.leaves import init_parameter
from spflow.utils.projections import proj_convex_to_real, proj_real_to_convex
from spflow.utils.sampling_context import SIMPLE


class HistogramDist:
    """Piecewise-constant histogram density with fixed bin edges.

    This distribution models a continuous univariate random variable as a
    piecewise-constant density over fixed bin edges. It mirrors the subset of
    the ``torch.distributions`` interface required by :class:`~spflow.modules.leaves.leaf.LeafModule`:
    ``log_prob``, ``sample``, and ``mode``.

    Shape conventions:
        - Parameters have batch shape ``(F, C, R, B)`` where:
          ``F``=features (must be 1 for Histogram leaves), ``C``=channels,
          ``R``=repetitions, ``B``=number of bins.
        - ``sample((N,))`` returns ``(N, F, C, R)``.
        - ``log_prob(x)`` returns ``(N, F, C, R)``.
    """

    def __init__(self, *, bin_edges: Tensor, logits: Tensor, min_prob: float = 1e-12) -> None:
        if bin_edges.dim() != 1:
            raise InvalidParameterError(f"bin_edges must be 1D, got shape {tuple(bin_edges.shape)}.")
        if bin_edges.numel() < 2:
            raise InvalidParameterError("bin_edges must contain at least two edges.")
        if not torch.isfinite(bin_edges).all():
            raise InvalidParameterError("bin_edges must be finite.")
        if not torch.all(bin_edges[1:] > bin_edges[:-1]):
            raise InvalidParameterError("bin_edges must be strictly increasing.")

        if logits.dim() != 4:
            raise InvalidParameterError(f"logits must be 4D (F,C,R,B), got shape {tuple(logits.shape)}.")

        self._bin_edges = bin_edges
        self._logits = logits
        self._min_prob = float(min_prob)

    @property
    def bin_edges(self) -> Tensor:
        return self._bin_edges

    @property
    def nbins(self) -> int:
        return int(self._bin_edges.numel() - 1)

    @property
    def probs(self) -> Tensor:
        return torch.softmax(self._logits, dim=-1)

    @property
    def _bin_widths(self) -> Tensor:
        return self._bin_edges[1:] - self._bin_edges[:-1]

    @property
    def _bin_midpoints(self) -> Tensor:
        return (self._bin_edges[:-1] + self._bin_edges[1:]) / 2.0

    @property
    def _bin_densities(self) -> Tensor:
        widths = self._bin_widths
        probs = self.probs
        densities = probs / rearrange(widths, "b -> 1 1 1 b")
        return densities

    @property
    def mode(self) -> Tensor:
        """Return mode as bin midpoint of the maximum-probability bin."""
        max_idx = torch.argmax(self.probs, dim=-1)  # (F, C, R)
        mids = self._bin_midpoints.to(device=max_idx.device, dtype=self._logits.dtype)  # (B,)
        return mids[max_idx]  # (F, C, R)

    def _align_x(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            return rearrange(x, "n f -> n f 1 1")
        if x.dim() == 3:
            return rearrange(x, "n f c -> n f c 1")
        if x.dim() == 4:
            return x
        raise InvalidParameterError(
            f"Expected x to have shape (N,F), (N,F,C), or (N,F,C,R); got {tuple(x.shape)}."
        )

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability density.

        Values outside the support ``[bin_edges[0], bin_edges[-1])`` receive ``-inf``.
        """
        x = self._align_x(x)
        n_samples, num_features, _, _ = x.shape

        if num_features != self._logits.shape[0]:
            raise InvalidParameterError(
                f"Feature mismatch: x has {num_features} features but logits have {self._logits.shape[0]}."
            )

        x_broadcast = torch.broadcast_to(
            x, (n_samples, *self._logits.shape[:-1])
        ).contiguous()  # (N, F, C, R)

        edges = self._bin_edges.to(device=x_broadcast.device, dtype=x_broadcast.dtype)
        bin_idx = torch.bucketize(x_broadcast, edges, right=True) - 1  # (N, F, C, R)

        in_support = torch.isfinite(x_broadcast) & (x_broadcast >= edges[0]) & (x_broadcast < edges[-1])

        bin_idx_safe = bin_idx.clamp(0, self.nbins - 1)
        densities = self._bin_densities.to(device=x_broadcast.device, dtype=x_broadcast.dtype)  # (F,C,R,B)
        densities = repeat(rearrange(densities, "f c r b -> 1 f c r b"), "1 f c r b -> n f c r b", n=n_samples)
        gathered = rearrange(
            densities.gather(-1, rearrange(bin_idx_safe, "n f c r -> n f c r 1")),
            "n f c r 1 -> n f c r",
        )

        min_density = self._min_prob / self._bin_widths.max().to(device=gathered.device, dtype=gathered.dtype)
        log_p = torch.log(gathered.clamp_min(min_density))
        log_p = torch.where(in_support, log_p, x_broadcast.new_full((), float("-inf")))
        return log_p

    def sample(self, sample_shape: torch.Size | tuple[int, ...]) -> Tensor:
        """Sample values, uniformly within sampled bins."""
        if isinstance(sample_shape, torch.Size):
            n_samples = int(sample_shape[0]) if len(sample_shape) else 1
        else:
            n_samples = int(sample_shape[0]) if len(sample_shape) else 1

        probs = self.probs
        f, c, r, b = probs.shape
        probs_flat = probs.reshape(-1, b)  # (F*C*R, B)

        cat = torch.distributions.Categorical(probs=probs_flat)
        bin_idx = cat.sample((n_samples,))  # (N, F*C*R)

        edges = self._bin_edges.to(device=bin_idx.device, dtype=self._logits.dtype)
        left = edges[bin_idx]
        right = edges[bin_idx + 1]

        u = torch.rand_like(left)
        x = left + u * (right - left)
        return x.reshape(n_samples, f, c, r)


class HistogramDistWithDifferentiableSampling(HistogramDist):
    """Histogram distribution with differentiable sampling via SIMPLE over bins."""

    has_rsample = True

    def sample(self, sample_shape: torch.Size | tuple[int, ...]) -> Tensor:
        return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size | tuple[int, ...]) -> Tensor:
        if isinstance(sample_shape, torch.Size):
            n_samples = int(sample_shape[0]) if len(sample_shape) else 1
        else:
            n_samples = int(sample_shape[0]) if len(sample_shape) else 1

        logits = self._logits.expand(n_samples, *self._logits.shape)  # (N, F, C, R, B)
        samples_oh = SIMPLE(logits=logits, dim=-1, is_mpe=False)

        edges = self._bin_edges.to(device=logits.device, dtype=logits.dtype)
        left_edges = edges[:-1]
        right_edges = edges[1:]

        left = (samples_oh * rearrange(left_edges, "b -> 1 1 1 1 b")).sum(dim=-1)
        right = (samples_oh * rearrange(right_edges, "b -> 1 1 1 1 b")).sum(dim=-1)

        u = torch.rand_like(left)
        return left + u * (right - left)

class Histogram(LeafModule):
    """Histogram leaf distribution (continuous piecewise-constant density).

    The histogram uses fixed bin edges and learnable bin probabilities (stored as logits).
    Within each bin, the density is constant: ``p(bin) / width(bin)``.

    Notes:
        - This leaf is **univariate**: ``len(scope.query) == 1``.
        - Values outside the support ``[bin_edges[0], bin_edges[-1])`` have log-likelihood ``-inf``.
        - NaN values are marginalized out by :meth:`~spflow.modules.leaves.leaf.LeafModule.log_likelihood`
          and contribute ``0`` to the log-likelihood.
        - MPE returns the midpoint of the maximum-probability bin (per channel/repetition).
    """

    def __init__(
        self,
        scope: Scope,
        *,
        bin_edges: Tensor,
        out_channels: int = 1,
        num_repetitions: int = 1,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        min_prob: float = 1e-12,
        validate_args: bool | None = True,
    ) -> None:
        if probs is not None and logits is not None:
            raise InvalidParameterCombinationError("Histogram accepts either probs or logits, not both.")

        if len(scope.query) != 1:
            raise InvalidParameterError(
                "Histogram leaf is univariate and requires scope with exactly one query RV."
            )

        bin_edges = torch.as_tensor(bin_edges, dtype=torch.float32)
        if bin_edges.dim() != 1:
            raise InvalidParameterError(f"bin_edges must be 1D, got shape {tuple(bin_edges.shape)}.")
        if bin_edges.numel() < 2:
            raise InvalidParameterError("bin_edges must contain at least two edges.")
        if not torch.isfinite(bin_edges).all():
            raise InvalidParameterError("bin_edges must be finite.")
        if not torch.all(bin_edges[1:] > bin_edges[:-1]):
            raise InvalidParameterError("bin_edges must be strictly increasing.")

        self._min_prob = float(min_prob)
        param_source = logits if logits is not None else probs

        super().__init__(
            scope=scope,
            out_channels=out_channels,  # type: ignore[arg-type]
            num_repetitions=num_repetitions,
            params=[param_source],
            validate_args=validate_args,
        )

        nbins = int(bin_edges.numel() - 1)
        if param_source is not None and int(param_source.shape[-1]) != nbins:
            raise InvalidParameterError(
                f"Last dim of probs/logits must match nbins={nbins}, got {int(param_source.shape[-1])}."
            )

        self.register_buffer("bin_edges", torch.empty(size=[]))
        self.bin_edges = bin_edges

        param_shape = (*self._event_shape, nbins)
        init_value = init_parameter(
            param=param_source,
            event_shape=param_shape,
            init=lambda shape: torch.rand(shape).softmax(dim=-1),
        )

        logits_tensor = init_value if logits is not None else proj_convex_to_real(init_value)
        self._logits = nn.Parameter(logits_tensor)

    @property
    def logits(self) -> Tensor:
        """Unconstrained logits parameterizing bin probabilities."""
        return self._logits

    @logits.setter
    def logits(self, value: Tensor) -> None:
        value_tensor = torch.as_tensor(value, dtype=self._logits.dtype, device=self._logits.device)
        self._logits.data = value_tensor

    @property
    def probs(self) -> Tensor:
        """Bin probabilities in natural space (softmax of logits)."""
        return proj_real_to_convex(self._logits)

    @probs.setter
    def probs(self, value: Tensor) -> None:
        value_tensor = torch.as_tensor(value, dtype=self._logits.dtype, device=self._logits.device)
        if not torch.isfinite(value_tensor).all():
            raise InvalidParameterError("probs must be finite.")
        if (value_tensor < 0).any():
            raise InvalidParameterError("probs must be non-negative.")
        value_tensor = value_tensor / value_tensor.sum(dim=-1, keepdim=True).clamp_min(self._min_prob)
        self._logits.data = proj_convex_to_real(value_tensor.clamp_min(self._min_prob))

    @property
    def _torch_distribution_class(self):
        """Histogram uses a custom distribution, not a torch.distributions class."""
        return None

    def distribution(self, with_differentiable_sampling: bool = False) -> HistogramDist:
        dist_cls = HistogramDistWithDifferentiableSampling if with_differentiable_sampling else HistogramDist
        return dist_cls(
            bin_edges=self.bin_edges.to(self._logits.device), logits=self._logits, min_prob=self._min_prob
        )

    @property
    def _supported_value(self) -> Tensor:
        """Value in support used for NaN imputation prior to marginalization."""
        return (self.bin_edges[0] + self.bin_edges[-1]) / 2.0

    def params(self) -> dict[str, Tensor]:
        return {"logits": self.logits}

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> dict[str, Tensor]:
        del bias_correction
        # data: (N, F=1, 1, 1)
        x = rearrange(data, "n f 1 1 -> n f")
        w = weights  # (N, F, C, R)
        if x.dim() != 2 or x.shape[1] != 1:
            raise InvalidParameterError(
                f"Histogram expects univariate scoped data, got shape {tuple(x.shape)}."
            )

        edges = self.bin_edges.to(device=x.device, dtype=x.dtype)
        nbins = int(edges.numel() - 1)
        bin_idx = torch.bucketize(x, edges, right=True) - 1  # (N, 1)

        in_support = torch.isfinite(x) & (x >= edges[0]) & (x < edges[-1])
        if not in_support.all():
            raise InvalidParameterError("MLE data contains values outside histogram support.")

        bin_idx = rearrange(bin_idx, "n 1 -> n")
        w_flat = rearrange(w[:, 0], "n c r -> n (c r)")
        one_hot = torch.nn.functional.one_hot(bin_idx, nbins).to(dtype=w_flat.dtype)  # (N, B)

        counts = w_flat.transpose(0, 1) @ one_hot  # (C*R, B)
        probs_est = counts / counts.sum(dim=-1, keepdim=True).clamp_min(self._min_prob)
        probs_est = probs_est.clamp_min(self._min_prob)
        probs_est = probs_est / probs_est.sum(dim=-1, keepdim=True)

        probs_est = rearrange(
            probs_est,
            "(c r) b -> 1 c r b",
            c=self.out_shape.channels,
            r=self.out_shape.repetitions,
        )
        return {"probs": probs_est}

    def marginalize(self, marg_rvs: list[int], prune: bool = True, cache=None):
        del prune, cache
        if self.is_conditional:
            raise RuntimeError(
                f"Marginalization not supported for conditional leaf {self.__class__.__name__}."
            )

        if any(rv in marg_rvs for rv in self.scope.query):
            return None

        return Histogram(
            scope=self.scope.copy(), bin_edges=self.bin_edges.detach().clone(), logits=self.logits.detach()
        )
