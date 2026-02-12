"""PIC (Probabilistic Integral Circuits) construction and materialization.

This module implements the core pipeline from the NeurIPS 2024 paper:
“Scaling Continuous Latent Variable Models as Probabilistic Integral Circuits”:

- Algorithm 1: RG → PIC
- Algorithm 2: merge (Tucker / CP)
- Algorithm 3: PIC → QPC (tensorized circuit materialization)

The PIC nodes in this file are symbolic: they do not support direct inference.
Inference is performed on the materialized QPC via standard SPFlow modules.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Callable, Dict, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterError, ShapeError, StructureError
from spflow.meta.data.scope import Scope
from spflow.meta.region_graph import Region, RegionGraph
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.ops.cat import Cat
from spflow.modules.products.elementwise_product import ElementwiseProduct
from spflow.modules.products.outer_product import OuterProduct
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from spflow.zoo.pic.integral import Integral
from spflow.zoo.pic.weighted_sum import WeightedSum
from spflow.zoo.pic.functional_sharing import FunctionGroup


class MergeStrategy(Enum):
    """Merge strategy for RG → PIC (Algorithm 2).

    - AUTO: Tucker if Z_u1 != Z_u2 else CP (paper semantics)
    - TUCKER: always Tucker merge
    - CP: always CP merge (requires Z_u1 == Z_u2)
    """

    AUTO = "auto"
    TUCKER = "tucker"
    CP = "cp"


@dataclasses.dataclass(frozen=True)
class QuadratureRule:
    """1D quadrature rule used for all PIC latents."""

    points: Tensor  # (K,)
    weights: Tensor  # (K,)


class PICInput(Protocol):
    """Protocol for PIC input units (leaf regions).

    A PIC input unit represents a positive function f_u(X_u, Z_u). During materialization
    (Algorithm 3, Line 4), it becomes a QPC input layer that outputs a vector indexed by
    the quadrature points of Z_u.
    """

    scope: Scope
    latent_scope: Scope

    def materialize(self, quadrature_rule: QuadratureRule) -> Module:
        """Materialize this input unit to a QPC leaf/module.

        The returned module must output `K^{|Z_u|}` channels and be compatible with SPFlow
        log-likelihood computation.
        """


def _symbolic_feature_to_scope(scope: Scope) -> np.ndarray:
    """Helper to create univariate feature_to_scope mapping for symbolic PIC nodes."""
    return np.array([Scope([rv]) for rv in scope.query])


class PICSum(Module):
    """Symbolic PIC sum unit +([u_i, w_i])."""

    def __init__(self, inputs: Sequence[Module], weights: Tensor, latent_scope: Scope) -> None:
        super().__init__()
        if len(inputs) == 0:
            raise InvalidParameterError("PICSum requires at least one input.")

        if weights.ndim != 1 or weights.shape[0] != len(inputs):
            raise ShapeError(f"PICSum weights must have shape ({len(inputs)},), got {tuple(weights.shape)}.")
        if not torch.all(weights > 0):
            raise InvalidParameterError("PICSum weights must be strictly positive.")
        if not torch.allclose(weights.sum(), weights.new_tensor(1.0)):
            raise InvalidParameterError("PICSum weights must sum to 1.")

        self.inputs = nn.ModuleList(list(inputs))
        self.weights = weights
        self.latent_scope = latent_scope

        # All children represent the same observed region X.
        if not Scope.all_equal([m.scope for m in self.inputs]):  # type: ignore[attr-defined]
            raise StructureError("All PICSum children must have identical observed scope.")
        self.scope = self.inputs[0].scope  # type: ignore[attr-defined]

        # Symbolic shape placeholder.
        self.in_shape = ModuleShape(features=len(self.scope.query), channels=1, repetitions=1)
        self.out_shape = ModuleShape(features=len(self.scope.query), channels=1, repetitions=1)

    @property
    def feature_to_scope(self) -> np.ndarray:
        # Best-effort mapping; PIC nodes are not used for inference.
        return _symbolic_feature_to_scope(self.scope)

    def log_likelihood(self, data: Tensor, cache=None) -> Tensor:  # pragma: no cover
        raise NotImplementedError("PICSum is symbolic; materialize to QPC with pic2qpc().")

    def sample(
        self, num_samples=None, data=None, is_mpe: bool = False, cache=None, sampling_ctx=None
    ) -> Tensor:  # pragma: no cover
        raise NotImplementedError("PICSum is symbolic; materialize to QPC with pic2qpc().")

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> Tensor:  # pragma: no cover
        del data
        del sampling_ctx
        del cache
        del is_mpe
        raise NotImplementedError("PICSum is symbolic; materialize to QPC with pic2qpc().")

    def marginalize(
        self, marg_rvs: list[int], prune: bool = True, cache=None
    ) -> Module | None:  # pragma: no cover
        raise NotImplementedError("PICSum is symbolic; materialize to QPC with pic2qpc().")


class PICProduct(Module):
    """Symbolic PIC product unit ×([u1, u2])."""

    def __init__(self, left: Module, right: Module) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.inputs = nn.ModuleList([left, right])

        # Observed scopes must be disjoint (region partitions).
        if not left.scope.isdisjoint(right.scope):  # type: ignore[attr-defined]
            raise StructureError(f"PICProduct requires disjoint scopes, got {left.scope} and {right.scope}.")
        self.scope = left.scope.join(right.scope)  # type: ignore[attr-defined]

        z_left = getattr(left, "latent_scope", Scope([]))
        z_right = getattr(right, "latent_scope", Scope([]))
        self.latent_scope = Scope.join_all([z_left, z_right])

        # Symbolic shape placeholder.
        self.in_shape = ModuleShape(features=len(self.scope.query), channels=1, repetitions=1)
        self.out_shape = ModuleShape(features=len(self.scope.query), channels=1, repetitions=1)

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([Scope([rv]) for rv in self.scope.query])

    def log_likelihood(self, data: Tensor, cache=None) -> Tensor:  # pragma: no cover
        raise NotImplementedError("PICProduct is symbolic; materialize to QPC with pic2qpc().")

    def sample(
        self, num_samples=None, data=None, is_mpe: bool = False, cache=None, sampling_ctx=None
    ) -> Tensor:  # pragma: no cover
        raise NotImplementedError("PICProduct is symbolic; materialize to QPC with pic2qpc().")

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> Tensor:  # pragma: no cover
        del data
        del sampling_ctx
        del cache
        del is_mpe
        raise NotImplementedError("PICProduct is symbolic; materialize to QPC with pic2qpc().")

    def marginalize(
        self, marg_rvs: list[int], prune: bool = True, cache=None
    ) -> Module | None:  # pragma: no cover
        raise NotImplementedError("PICProduct is symbolic; materialize to QPC with pic2qpc().")


def rg2pic(
    rg: RegionGraph,
    *,
    merge_strategy: MergeStrategy = MergeStrategy.AUTO,
    leaf_factory: Callable[[Scope, Scope], Module],
    function_factory: Optional[Callable[[int, int], nn.Module]] = None,
    sum_weights_factory: Optional[Callable[[int], Tensor]] = None,
    integral_group_factory: Optional[Callable[[int, int, int], FunctionGroup]] = None,
) -> Module:
    """Algorithm 1: Convert RegionGraph to a symbolic PIC.

    Args:
        rg: Input RegionGraph (currently expected to be binary).
        merge_strategy: Merge rule (AUTO matches the paper).
        leaf_factory: Factory for leaf-region input units. Called with `(x_scope, z_scope)`.
        function_factory: Factory for integral functions f_u, called as `(z_dim, y_dim)`.
        sum_weights_factory: Optional factory for + weights of length N partitions (defaults to uniform).
        integral_group_factory: Optional factory for depth-wise functional sharing groups. If provided,
            integrals created at the same RG depth with the same `(z_dim, y_dim)` can share a group.

    Returns:
        Root PIC module (symbolic).
    """
    if leaf_factory is None:
        raise InvalidParameterError("leaf_factory must be provided.")

    # NOTE: The NeurIPS 2024 paper assumes binary partitions w.l.o.g., but some region graphs
    # (e.g., quad trees) use n-ary partitions. The symbolic PIC implementation in this file
    # currently only supports binary partitions. For tensorized QPC materialization we keep the
    # original n-ary RegionGraph attached to the returned PIC root (see below).

    # Allocate latent ids after observed RV ids.
    max_obs = max(rg.root.scope.query) if len(rg.root.scope.query) > 0 else 0
    next_latent_id = max_obs + 1

    def alloc_latent() -> int:
        nonlocal next_latent_id
        z = next_latent_id
        next_latent_id += 1
        return z

    # Compute RG depths (leaves at depth 0).
    depth: Dict[Region, int] = {}
    for region in rg.post_order():
        if not region.children:
            depth[region] = 0
        else:
            depth[region] = 1 + max(depth[ch] for part in region.children for ch in part)

    U: Dict[Region, Module] = {}

    # Optional depth-wise groups: key (depth, z_dim, y_dim) -> FunctionGroup
    integral_groups: Dict[Tuple[int, int, int], FunctionGroup] = {}

    for X in rg.post_order():
        if not X.children:
            z_scope = Scope([]) if X == rg.root else Scope([alloc_latent()])
            unit = leaf_factory(X.scope, z_scope)
            if not hasattr(unit, "latent_scope"):
                setattr(unit, "latent_scope", z_scope)
            if getattr(unit, "latent_scope") != z_scope:
                raise StructureError("leaf_factory must set latent_scope consistently with provided z_scope.")
            U[X] = unit
            continue

        merged_units: list[Module] = []
        rho = X == rg.root
        for partition in X.children:
            if len(partition) < 2:
                raise StructureError("A region partition must contain at least 2 child regions.")

            # The paper assumes binary partitions w.l.o.g. For n-ary partitions we binarize
            # locally by folding left-to-right. Tensorized QPC materialization can still use the
            # original n-ary RegionGraph via `pic2qpc(..., mode='tensorized')`.
            units = [U[ch] for ch in partition]
            current = units[0]
            for i, nxt in enumerate(units[1:], start=1):
                is_last = i == (len(units) - 1)
                current = _merge_units(
                    u1=current,
                    u2=nxt,
                    rho=rho if is_last else False,
                    merge_strategy=merge_strategy,
                    function_factory=function_factory,
                    alloc_latent=alloc_latent,
                    depth=depth[X],
                    integral_group_factory=integral_group_factory,
                    integral_groups=integral_groups,
                )
            merged_units.append(current)

        if len(merged_units) == 1:
            U[X] = merged_units[0]
        else:
            if sum_weights_factory is None:
                weights = torch.full((len(merged_units),), 1.0 / len(merged_units))
            else:
                weights = sum_weights_factory(len(merged_units))
            z_scope = getattr(merged_units[0], "latent_scope", Scope([]))
            if any(getattr(m, "latent_scope", Scope([])) != z_scope for m in merged_units[1:]):
                raise StructureError("All partitions of a region must yield the same latent_scope.")
            U[X] = PICSum(inputs=merged_units, weights=weights, latent_scope=z_scope)

    root = U[rg.root]

    # Attach build metadata so `pic2qpc(..., mode="tensorized")` can build a folded tensorized
    # QPC directly from the RegionGraph (including n-ary partitions).
    setattr(root, "_region_graph", rg)
    setattr(root, "_merge_strategy", merge_strategy)

    return root


def _merge_units(
    *,
    u1: Module,
    u2: Module,
    rho: bool,
    merge_strategy: MergeStrategy,
    function_factory: Optional[Callable[[int, int], nn.Module]],
    alloc_latent: Callable[[], int],
    depth: int,
    integral_group_factory: Optional[Callable[[int, int, int], FunctionGroup]],
    integral_groups: Dict[Tuple[int, int, int], FunctionGroup],
) -> Module:
    z1: Scope = getattr(u1, "latent_scope", Scope([]))
    z2: Scope = getattr(u2, "latent_scope", Scope([]))

    if merge_strategy == MergeStrategy.AUTO:
        chosen = MergeStrategy.CP if z1 == z2 else MergeStrategy.TUCKER
    else:
        chosen = merge_strategy

    if chosen == MergeStrategy.TUCKER:
        prod = PICProduct(u1, u2)
        y_scope = prod.latent_scope
        z_scope = Scope([]) if rho else Scope([alloc_latent()])

        z_dim = len(z_scope.query)
        y_dim = len(y_scope.query)
        fn = function_factory(z_dim, y_dim) if function_factory else None
        integral = Integral(
            input_module=prod,
            latent_scope=z_scope,
            integrated_latent_scope=y_scope,
            function=fn,
        )

        _maybe_attach_function_group(
            integral=integral,
            depth=depth,
            z_dim=z_dim,
            y_dim=y_dim,
            integral_group_factory=integral_group_factory,
            integral_groups=integral_groups,
        )
        return integral

    if chosen == MergeStrategy.CP:
        if z1 != z2:
            raise StructureError("CP-merge requires Z_u1 == Z_u2.")
        if len(z1.query) == 0:
            raise StructureError("CP-merge requires a non-empty shared latent scope.")

        z_new = Scope([]) if rho else Scope([alloc_latent()])
        z_dim = len(z_new.query)
        y_dim = len(z1.query)

        fn1 = function_factory(z_dim, y_dim) if function_factory else None
        fn2 = function_factory(z_dim, y_dim) if function_factory else None

        i1 = Integral(
            input_module=u1,
            latent_scope=z_new,
            integrated_latent_scope=z1,
            function=fn1,
        )
        i2 = Integral(
            input_module=u2,
            latent_scope=z_new,
            integrated_latent_scope=z2,
            function=fn2,
        )

        _maybe_attach_function_group(
            integral=i1,
            depth=depth,
            z_dim=z_dim,
            y_dim=y_dim,
            integral_group_factory=integral_group_factory,
            integral_groups=integral_groups,
        )
        _maybe_attach_function_group(
            integral=i2,
            depth=depth,
            z_dim=z_dim,
            y_dim=y_dim,
            integral_group_factory=integral_group_factory,
            integral_groups=integral_groups,
        )
        return PICProduct(i1, i2)

    raise InvalidParameterError(f"Unknown merge strategy: {merge_strategy}.")


def _maybe_attach_function_group(
    *,
    integral: Integral,
    depth: int,
    z_dim: int,
    y_dim: int,
    integral_group_factory: Optional[Callable[[int, int, int], FunctionGroup]],
    integral_groups: Dict[Tuple[int, int, int], FunctionGroup],
) -> None:
    if integral_group_factory is None:
        return
    if z_dim == 0:
        # Root integrals are typically unique and output scalar; skip grouping.
        return

    key = (depth, z_dim, y_dim)
    group = integral_groups.get(key)
    if group is None:
        group = integral_group_factory(depth, z_dim, y_dim)
        integral_groups[key] = group

    head_idx = group.add_unit(integral)
    integral.function = group
    integral.function_head_idx = head_idx


def pic2qpc(
    pic: Module,
    quadrature_rule: QuadratureRule,
    *,
    mode: str = "expanded",
    tensorized_config: "TensorizedQPCConfig" | None = None,
) -> Module:
    """Algorithm 3: Materialize a symbolic PIC into a QPC.

    This function supports two modes:

    - `mode="expanded"`: current exact materialization into a standard SPFlow module graph
      (OuterProduct / ElementwiseProduct / WeightedSum), matching Eqs. (3)–(4) in the paper.
    - `mode="tensorized"`: folded/tensorized materialization inspired by the authors' reference
      implementation (see `reference-repos/ten-pics`). This returns a single SPFlow `Module`
      that evaluates the folded tensorized circuit and parameterizes its inner layers via
      neural functional sharing (perm_dim/norm_dim + chunked quadrature evaluation).
    """
    if mode not in {"expanded", "tensorized"}:
        raise InvalidParameterError(f"Unknown pic2qpc mode: {mode!r}.")

    if mode == "tensorized":
        if tensorized_config is None:
            raise InvalidParameterError("tensorized_config must be provided when mode='tensorized'.")
        if not hasattr(pic, "_region_graph"):
            raise StructureError(
                "Tensorized materialization requires the PIC root to have `_region_graph` attached. "
                "Build the PIC via `rg2pic(...)` or attach the RegionGraph manually."
            )

        from spflow.zoo.pic.tensorized.qpc import TensorizedQPC

        rg = getattr(pic, "_region_graph")
        if not isinstance(rg, RegionGraph):
            raise StructureError("PIC `_region_graph` must be a `spflow.meta.region_graph.RegionGraph`.")
        return TensorizedQPC.from_region_graph(rg, quadrature_rule=quadrature_rule, config=tensorized_config)

    # Default: expanded materialization.
    if quadrature_rule.points.ndim != 1 or quadrature_rule.weights.ndim != 1:
        raise ShapeError("QuadratureRule points and weights must be 1D tensors.")
    if quadrature_rule.points.shape[0] != quadrature_rule.weights.shape[0]:
        raise ShapeError("QuadratureRule points and weights must have the same length.")
    if not torch.all(quadrature_rule.weights >= 0):
        raise InvalidParameterError("Quadrature weights must be non-negative.")

    K = int(quadrature_rule.points.shape[0])
    L: Dict[Module, Module] = {}

    def k_pow(scope: Scope) -> int:
        return K ** len(scope.query)

    def grid_points(num_dims: int, *, device, dtype) -> Tensor:
        """Return a (K^num_dims, num_dims) tensor of quadrature point assignments."""
        if num_dims == 0:
            return torch.empty((1, 0), device=device, dtype=dtype)
        meshes = torch.meshgrid(
            *[quadrature_rule.points.to(device=device, dtype=dtype) for _ in range(num_dims)],
            indexing="ij",
        )
        stacked = torch.stack(meshes, dim=-1)  # (K, ..., K, num_dims)
        return rearrange(stacked, "... d -> (...) d")

    def kron_weights(num_dims: int, *, device, dtype) -> Tensor:
        """Return Kronecker-product quadrature weights of length K^num_dims."""
        if num_dims == 0:
            return torch.ones((1,), device=device, dtype=dtype)
        w = quadrature_rule.weights.to(device=device, dtype=dtype)
        out = w
        for _ in range(num_dims - 1):
            out = torch.kron(out, w)
        return out

    def materialize_integral_group(group: FunctionGroup) -> None:
        # Collect only Integral units.
        units = [u for u in group.units if isinstance(u, Integral)]
        if len(units) == 0:
            return

        # Ensure children compiled.
        for u in units:
            _visit(u.inputs)

        # Group by (out_dim, in_dim) because root/other integrals may differ.
        buckets: Dict[Tuple[int, int], list[Integral]] = {}
        for u in units:
            z_dim = len(u.latent_scope.query)
            y_dim = len(u.integrated_latent_scope.query)
            buckets.setdefault((z_dim, y_dim), []).append(u)

        for (z_dim, y_dim), bucket in buckets.items():
            if z_dim == 0:
                # Skip scalar-output integrals (typically root).
                continue

            device = bucket[0].inputs.device
            dtype = quadrature_rule.points.dtype

            z_grid = grid_points(z_dim, device=device, dtype=dtype)  # (K^z, z_dim)
            y_grid = grid_points(y_dim, device=device, dtype=dtype)  # (K^y, y_dim)

            out_ch = z_grid.shape[0]
            in_ch = y_grid.shape[0]

            # Broadcast into (out_ch, in_ch, dim).
            num_output_channels = out_ch
            num_input_channels = in_ch
            z = repeat(z_grid, "co zd -> co ci zd", ci=num_input_channels)
            y = repeat(y_grid, "ci yd -> co ci yd", co=num_output_channels)

            vals_all = group.evaluate_batched(z, y)  # (num_heads, out_ch, in_ch)

            y_w = kron_weights(y_dim, device=device, dtype=dtype)  # (in_ch,)
            vals_all = vals_all * rearrange(y_w, "ci -> 1 1 ci")

            for u in bucket:
                head_idx = u.function_head_idx
                if head_idx is None:
                    continue
                if u in L:
                    continue
                child = L[u.inputs]
                if child.out_shape.channels != in_ch:
                    raise ShapeError(
                        f"Integral child channels mismatch: expected {in_ch}, got {child.out_shape.channels}."
                    )
                weights_mat = rearrange(vals_all[head_idx], "co ci -> ci co")
                weights_full = repeat(
                    rearrange(weights_mat, "ci co -> 1 ci co 1"),
                    "1 ci co 1 -> f ci co 1",
                    f=child.out_shape.features,
                )
                L[u] = WeightedSum(inputs=child, weights=weights_full)

    def _visit(u: Module) -> Module:
        if u in L:
            return L[u]

        # Materialize PIC input unit.
        if hasattr(u, "materialize") and hasattr(u, "latent_scope"):
            qpc_leaf = u.materialize(quadrature_rule)  # type: ignore[attr-defined]
            L[u] = qpc_leaf
            return qpc_leaf

        # Materialize PIC sum.
        if isinstance(u, PICSum):
            children = [_visit(ch) for ch in u.inputs]
            # All children must have the same Z_u output size.
            out_ch = children[0].out_shape.channels
            if any(ch.out_shape.channels != out_ch for ch in children):
                raise ShapeError("PICSum children must have identical channel count after materialization.")

            cat = Cat(inputs=children, dim=2)
            n = len(children)
            in_ch = n * out_ch

            # Eq. (4): W = ||_i w_i I_{K^|Z|}.
            W = torch.zeros((in_ch, out_ch), dtype=u.weights.dtype, device=cat.device)
            for i in range(n):
                start = i * out_ch
                W[start : start + out_ch, :] = (
                    torch.eye(out_ch, device=W.device, dtype=W.dtype) * u.weights[i]
                )

            weights_full = repeat(
                rearrange(W, "ci co -> 1 ci co 1"),
                "1 ci co 1 -> f ci co 1",
                f=cat.out_shape.features,
            )
            L[u] = WeightedSum(inputs=cat, weights=weights_full)
            return L[u]

        # Recurse for PIC product / integral.
        if isinstance(u, PICProduct):
            left = _visit(u.left)
            right = _visit(u.right)

            z_left = getattr(u.left, "latent_scope", Scope([]))
            z_right = getattr(u.right, "latent_scope", Scope([]))
            if z_left == z_right:
                L[u] = ElementwiseProduct(inputs=[left, right])
            else:
                L[u] = OuterProduct(inputs=[left, right])
            return L[u]

        if isinstance(u, Integral):
            # If the integral uses a function group, materialize the whole group at once.
            if isinstance(u.function, FunctionGroup):
                materialize_integral_group(u.function)
                if u in L:
                    return L[u]

            child = _visit(u.inputs)

            z_dim = len(u.latent_scope.query)
            y_dim = len(u.integrated_latent_scope.query)
            out_ch = k_pow(u.latent_scope)
            in_ch = k_pow(u.integrated_latent_scope)

            if child.out_shape.channels != in_ch:
                raise ShapeError(
                    f"Integral child channels mismatch: expected {in_ch}, got {child.out_shape.channels}."
                )
            if u.function is None:
                raise StructureError("Integral unit missing weighting function.")

            device = child.device
            dtype = quadrature_rule.points.dtype
            z_grid = grid_points(z_dim, device=device, dtype=dtype)
            y_grid = grid_points(y_dim, device=device, dtype=dtype)

            num_output_channels = out_ch
            num_input_channels = in_ch
            z = repeat(z_grid, "co zd -> co ci zd", ci=num_input_channels)
            y = repeat(y_grid, "ci yd -> co ci yd", co=num_output_channels)

            if isinstance(u.function, FunctionGroup):
                head = 0 if u.function.sharing_type == "f" else (u.function_head_idx or 0)
                vals = u.function.evaluate_batched(z, y)[head]
            else:
                vals = u.function(z, y)  # type: ignore[misc]

            y_w = kron_weights(y_dim, device=device, dtype=dtype)
            Wf = vals * rearrange(y_w, "ci -> 1 ci")  # (out_ch, in_ch)

            # WeightedSum expects (in_ch, out_ch) in its weights tensor.
            weights_mat = rearrange(Wf, "co ci -> ci co").contiguous()
            weights_full = repeat(
                rearrange(weights_mat, "ci co -> 1 ci co 1"),
                "1 ci co 1 -> f ci co 1",
                f=child.out_shape.features,
            )
            L[u] = WeightedSum(inputs=child, weights=weights_full)
            return L[u]

        raise StructureError(f"Unsupported PIC node type for materialization: {type(u)}.")

    return _visit(pic)
