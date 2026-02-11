"""Chow-Liu Tree (CLTree) multivariate discrete leaf.

Implements a discrete Chow-Liu tree distribution over a multivariate scope:

    P(x) = P(x_root) * Π_i P(x_i | x_parent(i))

Supports:
- exact log-likelihood with NaN evidence (marginalization via belief propagation)
- conditional sampling under evidence
- MPE completion under evidence (max-product)
- (weighted) MLE for CPTs, with optional one-time structure learning (Chow-Liu)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterError, ShapeError, UnsupportedOperationError
from spflow.meta import Scope
from spflow.modules.leaves.leaf import LeafModule
from spflow.utils.cache import Cache, cached
from spflow.utils.leaves import apply_nan_strategy


@dataclass(frozen=True)
class _TreeOrders:
    pre_order: tuple[int, ...]
    post_order: tuple[int, ...]


def _validate_discrete_values(data: Tensor, *, K: int) -> None:
    if data.numel() == 0:
        return

    finite = data[torch.isfinite(data)]
    if finite.numel() == 0:
        return

    if not torch.allclose(finite, finite.round()):
        raise InvalidParameterError("CLTree expects discrete integer values (or NaN for missing).")

    if finite.min() < 0 or finite.max() >= K:
        raise InvalidParameterError(f"CLTree observed values must be in {{0, ..., {K - 1}}}.")


def _compute_orders_from_parents(parents: list[int]) -> _TreeOrders:
    num_features = len(parents)
    if num_features == 0:
        return _TreeOrders((), ())

    roots = [i for i, p in enumerate(parents) if p == -1]
    if len(roots) != 1:
        raise InvalidParameterError(f"CLTree requires exactly one root (-1 parent), got {len(roots)}.")
    root = roots[0]

    children: list[list[int]] = [[] for _ in range(num_features)]
    for child, parent in enumerate(parents):
        if parent == -1:
            continue
        if parent < 0 or parent >= num_features:
            raise InvalidParameterError("CLTree parent index out of range.")
        children[parent].append(child)

    # Deterministic order: visit children sorted by index.
    pre: list[int] = []
    stack: list[int] = [root]
    while stack:
        node = stack.pop()
        pre.append(node)
        for ch in sorted(children[node], reverse=True):
            stack.append(ch)

    post = list(reversed(pre))
    return _TreeOrders(tuple(pre), tuple(post))


def _prim_maximum_spanning_tree(weights: Tensor, *, root: int = 0) -> list[int]:
    """Return parent pointers for a maximum spanning tree rooted at `root`.

    Args:
        weights: (F, F) symmetric weight matrix with zeros on diagonal.
        root: Root node index.

    Returns:
        Parent list of length F with parent[root] = -1.
    """
    if weights.dim() != 2 or weights.shape[0] != weights.shape[1]:
        raise ShapeError(f"Expected square weights matrix, got shape {tuple(weights.shape)}.")

    num_features = int(weights.shape[0])
    if num_features == 0:
        return []
    if root < 0 or root >= num_features:
        raise InvalidParameterError("Root index out of range.")

    in_tree = torch.zeros(num_features, dtype=torch.bool)
    best_w = torch.full((num_features,), float("-inf"), dtype=weights.dtype, device=weights.device)
    best_p = torch.full((num_features,), -1, dtype=torch.long, device=weights.device)

    in_tree[root] = True
    best_w[:] = weights[root]
    best_p[:] = root
    best_w[root] = float("-inf")
    best_p[root] = -1

    parents = [-1 for _ in range(num_features)]
    for _ in range(num_features - 1):
        cand_w = best_w.clone()
        cand_w[in_tree] = float("-inf")
        v = int(torch.argmax(cand_w).item())
        if cand_w[v].item() == float("-inf"):
            raise UnsupportedOperationError("CLTree structure learning failed: graph appears disconnected.")
        parents[v] = int(best_p[v].item())
        in_tree[v] = True

        # Update candidates with edges from v.
        improved = (~in_tree) & (weights[v] > best_w)
        best_w[improved] = weights[v][improved]
        best_p[improved] = v

    parents[root] = -1
    return parents


class CLTree(LeafModule):
    """Chow-Liu tree discrete multivariate leaf.

    Notes:
        - `log_likelihood()` supports NaNs by exact marginalization (belief propagation).
        - `maximum_likelihood_estimation()` learns structure once (if missing) and then
          updates CPTs. Structure is shared across channels/repetitions.
        - `sample()` supports conditional sampling and MPE completion under evidence.
    """

    def __init__(
        self,
        scope: Scope | int | list[int] | tuple[int, ...],
        out_channels: int = 1,
        num_repetitions: int = 1,
        *,
        K: int,
        alpha: float = 0.01,
        parents: Tensor | None = None,
        log_cpt: Tensor | None = None,
        validate_args: bool | None = True,
    ) -> None:
        if K < 2:
            raise InvalidParameterError(f"CLTree requires K >= 2, got {K}.")
        if alpha <= 0:
            raise InvalidParameterError(f"CLTree requires alpha > 0, got {alpha}.")

        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[log_cpt],
            parameter_fn=None,
            validate_args=validate_args,
        )

        if len(self.scope.query) < 2:
            raise InvalidParameterError("CLTree requires a scope with at least 2 variables.")

        self.K: int = int(K)
        self.alpha: float = float(alpha)

        if parents is not None:
            if parents.dim() != 1 or parents.shape[0] != self.out_shape.features:
                raise ShapeError(
                    f"parents must have shape ({self.out_shape.features},), got {tuple(parents.shape)}."
                )
            parents_list = [int(v) for v in parents.detach().cpu().tolist()]
            orders = _compute_orders_from_parents(parents_list)
            self.register_buffer("parents", parents.to(dtype=torch.long))
            self.register_buffer("pre_order", torch.tensor(orders.pre_order, dtype=torch.long))
            self.register_buffer("post_order", torch.tensor(orders.post_order, dtype=torch.long))
        else:
            self.register_buffer("parents", torch.full((self.out_shape.features,), -1, dtype=torch.long))
            self.register_buffer("pre_order", torch.arange(self.out_shape.features, dtype=torch.long))
            self.register_buffer(
                "post_order",
                torch.arange(self.out_shape.features - 1, -1, -1, dtype=torch.long),
            )

        if log_cpt is None:
            init = torch.rand(
                self.out_shape.features, self.out_shape.channels, self.out_shape.repetitions, K, K
            )
            init = init / init.sum(dim=-2, keepdim=True).clamp_min(1e-12)
            log_cpt = init.clamp_min(1e-12).log()
        else:
            expected = (
                self.out_shape.features,
                self.out_shape.channels,
                self.out_shape.repetitions,
                K,
                K,
            )
            if tuple(log_cpt.shape) != expected:
                raise ShapeError(f"log_cpt must have shape {expected}, got {tuple(log_cpt.shape)}.")

        self.log_cpt = nn.Parameter(log_cpt)

    @property
    def _supported_value(self) -> float:
        return 0.0

    @property
    def _torch_distribution_class(self) -> type[torch.distributions.Distribution]:
        raise UnsupportedOperationError(
            "CLTree does not expose a torch.distributions.Distribution. "
            "Use CLTree.log_likelihood() / sample() instead."
        )

    def params(self) -> dict[str, Tensor]:
        return {"log_cpt": self.log_cpt}

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> dict[str, Tensor]:
        raise UnsupportedOperationError(
            "CLTree does not use LeafModule's template MLE hooks. "
            "Call CLTree.maximum_likelihood_estimation() instead."
        )

    def _has_learned_structure(self) -> bool:
        # Exactly one root required and parents must be within range for non-root.
        parents = self.parents.detach().cpu().tolist()
        roots = sum(1 for p in parents if p == -1)
        if roots != 1:
            return False
        for i, p in enumerate(parents):
            if p == -1:
                continue
            if p < 0 or p >= len(parents) or p == i:
                return False
        return True

    def _resolve_scoped_data(self, data: Tensor) -> Tensor:
        if data.dim() != 2:
            raise ShapeError(f"Data must be 2D (batch, num_features), got shape {tuple(data.shape)}.")
        scope_cols = self._resolve_scope_columns(num_features=data.shape[1])
        if len(scope_cols) != self.out_shape.features:
            raise ShapeError("Resolved scope columns do not match CLTree scope length.")
        return data[:, scope_cols]

    def _compute_mutual_information(self, x: Tensor, sample_weights: Tensor | None) -> Tensor:
        """Compute MI matrix (F, F) from data x (N, F) with values in {0..K-1}."""
        num_samples, num_features = x.shape
        K = self.K
        device = x.device
        dtype = torch.float64

        if sample_weights is None:
            w = torch.ones(num_samples, device=device, dtype=dtype)
        else:
            if sample_weights.dim() != 1 or sample_weights.shape[0] != num_samples:
                raise ShapeError(
                    f"sample_weights must have shape ({num_samples},), got {tuple(sample_weights.shape)}."
                )
            w = sample_weights.to(device=device, dtype=dtype)
            if not torch.isfinite(w).all():
                raise InvalidParameterError("sample_weights must be finite.")

        # Normalize weights to sum to N to match other MLE code paths.
        w = w * (num_samples / w.sum().clamp_min(1e-12))

        # Marginals: counts_i[f,k]
        counts_i = torch.full((num_features, K), self.alpha, dtype=dtype, device=device)
        for k in range(K):
            counts_i[:, k] += (w[:, None] * (x == k).to(dtype)).sum(dim=0)
        denom_i = counts_i.sum(dim=1, keepdim=True).clamp_min(1e-12)
        p_i = counts_i / denom_i
        log_p_i = p_i.clamp_min(1e-12).log()

        mi = torch.zeros((num_features, num_features), dtype=dtype, device=device)
        for i in range(num_features):
            for j in range(i + 1, num_features):
                counts_ij = torch.full((K, K), self.alpha, dtype=dtype, device=device)
                for a in range(K):
                    mask_a = (x[:, i] == a).to(dtype)
                    for b in range(K):
                        counts_ij[a, b] += (w * mask_a * (x[:, j] == b).to(dtype)).sum()
                p_ij = counts_ij / counts_ij.sum().clamp_min(1e-12)
                log_p_ij = p_ij.clamp_min(1e-12).log()
                # MI = sum p_ij * (log p_ij - log p_i - log p_j)
                mi_ij = (p_ij * (log_p_ij - log_p_i[i, :, None] - log_p_i[j, None, :])).sum()
                mi[i, j] = mi_ij
                mi[j, i] = mi_ij
        return mi

    def fit_structure(self, data: Tensor, *, sample_weights: Tensor | None = None) -> None:
        """Learn Chow-Liu structure from complete discrete data.

        Args:
            data: Tensor of shape (N, D) with values in {0..K-1} for this scope.
            sample_weights: Optional per-row weights (N,).
        """
        scoped = self._resolve_scoped_data(data)
        if torch.isnan(scoped).any():
            raise InvalidParameterError(
                "fit_structure requires complete data without NaNs (use nan_strategy='ignore')."
            )
        _validate_discrete_values(scoped, K=self.K)

        x = scoped.to(dtype=torch.long)
        mi = self._compute_mutual_information(x, sample_weights=sample_weights)
        parents_list = _prim_maximum_spanning_tree(mi.to(dtype=mi.dtype), root=0)
        orders = _compute_orders_from_parents(parents_list)

        self.parents.data = torch.tensor(parents_list, dtype=torch.long, device=self.parents.device)
        self.pre_order.data = torch.tensor(orders.pre_order, dtype=torch.long, device=self.pre_order.device)
        self.post_order.data = torch.tensor(
            orders.post_order, dtype=torch.long, device=self.post_order.device
        )

    @cached
    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:
        scoped = self._resolve_scoped_data(data)
        _validate_discrete_values(scoped, K=self.K)

        if not self._has_learned_structure():
            raise RuntimeError(
                "CLTree structure is not initialized. Call maximum_likelihood_estimation() or fit_structure() first."
            )

        N = scoped.shape[0]
        F, C, R = self.out_shape.features, self.out_shape.channels, self.out_shape.repetitions
        out = torch.zeros((N, F, C, R), dtype=self.log_cpt.dtype, device=self.log_cpt.device)

        parents = self.parents.tolist()
        pre_order = self.pre_order.tolist()
        post_order = self.post_order.tolist()
        root = parents.index(-1)

        # Fast path for complete data (no NaNs): index CPTs.
        if not torch.isnan(scoped).any():
            x = scoped.to(dtype=torch.long, device=self.log_cpt.device)
            log_cpt = self.log_cpt
            ll = torch.zeros((N, C, R), dtype=log_cpt.dtype, device=log_cpt.device)
            for i in range(F):
                p = parents[i]
                if p == -1:
                    table = rearrange(log_cpt[i, :, :, :, 0], "c r k -> k c r")
                    ll += table[x[:, i]]
                else:
                    table = rearrange(log_cpt[i], "c r k kp -> k kp c r")
                    ll += table[x[:, i], x[:, p]]
            out[:, 0, :, :] = ll
            return out

        # General path: belief propagation per sample.
        log_cpt = self.log_cpt
        K = self.K
        scoped_dev = scoped.to(device=log_cpt.device, dtype=log_cpt.dtype)

        for n in range(N):
            row = scoped_dev[n]

            # evidence masks
            is_obs = torch.isfinite(row)
            obs_val = torch.zeros((F,), dtype=torch.long, device=log_cpt.device)
            obs_val[is_obs] = row[is_obs].round().to(torch.long)

            # child_sum[node, c, r, x_node] = sum child->node messages conditioned on x_node
            child_sum = torch.zeros((F, C, R, K), dtype=log_cpt.dtype, device=log_cpt.device)

            for i in post_order:
                p = parents[i]
                if p == -1:
                    continue

                # score[c, r, x_i, x_p]
                score = log_cpt[i] + rearrange(child_sum[i], "c r k -> c r k 1")

                if bool(is_obs[i].item()):
                    xi = int(obs_val[i].item())
                    msg = score[:, :, xi, :]  # (C,R,K)
                else:
                    msg = torch.logsumexp(score, dim=2)  # (C,R,K)

                child_sum[p] += msg

            # Root marginal.
            root_score = log_cpt[root, :, :, :, 0] + child_sum[root]  # (C,R,K)
            if bool(is_obs[root].item()):
                xr = int(obs_val[root].item())
                ll = root_score[:, :, xr]  # (C,R)
            else:
                ll = torch.logsumexp(root_score, dim=2)  # (C,R)

            out[n, 0, :, :] = ll

        return out

    def _prepare_mle_inputs(
        self, data: Tensor, weights: Tensor | None, nan_strategy: str | None
    ) -> tuple[Tensor, Tensor]:
        scoped = self._resolve_scoped_data(data)

        if weights is None:
            weights = torch.ones(
                scoped.shape[0],
                self.out_shape.features,
                self.out_shape.channels,
                self.out_shape.repetitions,
                device=self.device,
            )
        if weights.dim() != 4:
            raise ShapeError(f"weights must be 4D (N,F,C,R), got shape {tuple(weights.shape)}.")
        if weights.shape[0] != scoped.shape[0]:
            raise ShapeError("weights batch dimension does not match data.")

        if nan_strategy is None:
            nan_strategy = "ignore"
        scoped, weights = apply_nan_strategy(nan_strategy, scoped, weights)
        if scoped.shape[0] < 2:
            raise InvalidParameterError("CLTree requires at least 2 complete samples for MLE.")

        _validate_discrete_values(scoped, K=self.K)
        return scoped, weights

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Tensor | None = None,
        bias_correction: bool = True,
        nan_strategy: str | None = "ignore",
        cache: Cache | None = None,
    ) -> None:
        del bias_correction, cache

        scoped, weights = self._prepare_mle_inputs(data=data, weights=weights, nan_strategy=nan_strategy)
        x = scoped.to(device=self.log_cpt.device, dtype=torch.long)

        # Learn structure once (shared across channels/repetitions).
        if not self._has_learned_structure():
            w_scalar = weights[:, 0].sum(dim=(1, 2)).detach()  # (N,)
            self.fit_structure(data=scoped, sample_weights=w_scalar)

        parents = self.parents.tolist()
        root = parents.index(-1)

        C, R, K = self.out_shape.channels, self.out_shape.repetitions, self.K
        log_cpt = torch.empty(
            (self.out_shape.features, C, R, K, K),
            dtype=self.log_cpt.dtype,
            device=self.log_cpt.device,
        )

        # Use weights from first feature (feature axis is a legacy of per-feature leaves).
        w = weights[:, 0].to(device=self.log_cpt.device, dtype=self.log_cpt.dtype)  # (N,C,R)

        # Root: P(x_root)
        counts_root = torch.full((C, R, K), self.alpha, dtype=self.log_cpt.dtype, device=self.log_cpt.device)
        for k in range(K):
            counts_root[:, :, k] += (w * (x[:, root] == k).to(w.dtype)[:, None, None]).sum(dim=0)
        probs_root = counts_root / counts_root.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        log_root = probs_root.clamp_min(1e-12).log()  # (C,R,K)
        log_cpt[root, :, :, :, :] = rearrange(log_root, "c r k -> c r k 1").expand(-1, -1, -1, K)

        # Conditionals: P(x_i | x_parent)
        for i in range(self.out_shape.features):
            p = parents[i]
            if p == -1:
                continue
            counts = torch.full(
                (C, R, K, K), self.alpha, dtype=self.log_cpt.dtype, device=self.log_cpt.device
            )
            for xp in range(K):
                mask_p = (x[:, p] == xp).to(w.dtype)[:, None, None]
                w_p = w * mask_p
                for xi in range(K):
                    counts[:, :, xi, xp] += (w_p * (x[:, i] == xi).to(w.dtype)[:, None, None]).sum(dim=0)
            probs = counts / counts.sum(dim=2, keepdim=True).clamp_min(1e-12)
            log_cpt[i] = probs.clamp_min(1e-12).log()

        with torch.no_grad():
            self.log_cpt.data = log_cpt

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx=None,
    ) -> Tensor:
        del cache
        data = self._prepare_sample_data(num_samples, data)

        if not self._has_learned_structure():
            raise RuntimeError(
                "CLTree structure is not initialized. Call maximum_likelihood_estimation() or fit_structure() first."
            )

        from spflow.utils.sampling_context import init_default_sampling_context

        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)

        scope_cols = self._resolve_scope_columns(num_features=data.shape[1])
        scoped = data[:, scope_cols]
        marg_mask = torch.isnan(scoped)

        ctx_channel_index, ctx_mask = self._slice_sampling_context(
            sampling_ctx=sampling_ctx, num_features=data.shape[1], scope_cols=scope_cols
        )
        samples_mask = marg_mask & ctx_mask

        instance_mask = samples_mask.any(dim=1)
        if not instance_mask.any():
            return data

        if sampling_ctx.repetition_idx is None:
            if self.out_shape.repetitions > 1:
                raise InvalidParameterError(
                    "Repetition index must be provided in sampling context for CLTree with multiple repetitions."
                )
            sampling_ctx.repetition_idx = torch.zeros(data.shape[0], dtype=torch.long, device=data.device)

        parents = self.parents.tolist()
        pre_order = self.pre_order.tolist()
        post_order = self.post_order.tolist()
        root = parents.index(-1)
        K = self.K

        # Ensure channel index is consistent across the CLTree scope for each row we sample.
        chosen_channels = ctx_channel_index[instance_mask]
        if not torch.all(chosen_channels == chosen_channels[:, :1]):
            raise InvalidParameterError(
                "CLTree requires a consistent channel_index across its scope per sample."
            )
        chan = chosen_channels[:, 0].to(torch.long)
        rep = sampling_ctx.repetition_idx[instance_mask].to(torch.long)

        scoped_vals = scoped[instance_mask].to(device=self.log_cpt.device, dtype=self.log_cpt.dtype)
        _validate_discrete_values(scoped_vals, K=K)

        for idx_in_batch, row_idx in enumerate(torch.where(instance_mask)[0].tolist()):
            c = int(chan[idx_in_batch].item())
            r = int(rep[idx_in_batch].item())
            row = scoped_vals[idx_in_batch]

            is_obs = torch.isfinite(row)
            obs_val = torch.zeros((self.out_shape.features,), dtype=torch.long, device=self.log_cpt.device)
            obs_val[is_obs] = row[is_obs].round().to(torch.long)

            child_sum = torch.zeros(
                (self.out_shape.features, K), dtype=self.log_cpt.dtype, device=self.log_cpt.device
            )
            backptr = torch.zeros((self.out_shape.features, K), dtype=torch.long, device=self.log_cpt.device)

            for i in post_order:
                p = parents[i]
                if p == -1:
                    continue

                score = self.log_cpt[i, c, r] + rearrange(child_sum[i], "k -> k 1")
                if bool(is_obs[i].item()):
                    xi = int(obs_val[i].item())
                    msg = score[xi]  # (K,)
                    backptr[i] = xi
                else:
                    if is_mpe:
                        best_x = torch.argmax(score, dim=0)  # (K,)
                        backptr[i] = best_x
                        msg = score.gather(dim=0, index=best_x.unsqueeze(0)).squeeze(0)  # (K,)
                    else:
                        msg = torch.logsumexp(score, dim=0)  # (K,)

                child_sum[p] += msg

            # Root posterior / choice.
            root_score = self.log_cpt[root, c, r, :, 0] + child_sum[root]  # (K,)
            if bool(is_obs[root].item()):
                xr = int(obs_val[root].item())
            else:
                if is_mpe:
                    xr = int(torch.argmax(root_score).item())
                else:
                    xr = int(torch.distributions.Categorical(logits=root_score).sample().item())

            assignment = torch.empty((self.out_shape.features,), dtype=torch.long, device=self.log_cpt.device)
            assignment[root] = xr

            for i in pre_order:
                if i == root:
                    continue
                p = parents[i]
                xp = int(assignment[p].item())
                if bool(is_obs[i].item()):
                    assignment[i] = int(obs_val[i].item())
                    continue
                if is_mpe:
                    assignment[i] = int(backptr[i, xp].item())
                else:
                    logits = self.log_cpt[i, c, r, :, xp] + child_sum[i]
                    assignment[i] = int(torch.distributions.Categorical(logits=logits).sample().item())

            # Write back only to requested mask positions.
            for j, col in enumerate(scope_cols):
                if bool(samples_mask[row_idx, j].item()):
                    data[row_idx, col] = assignment[j].to(dtype=data.dtype, device=data.device)

        return data
