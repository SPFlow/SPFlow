from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, cast

import numpy as np
import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.normal import Normal
from spflow.modules.module import Module
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, build_root_sampling_context
from spflow.zoo.sos.exp_socs import ExpSOCS
from spflow.zoo.sos.signed_categorical import SignedCategorical
from spflow.modules.sums.signed_sum import SignedSum
from spflow.modules.sos.socs import SOCS


@dataclass(frozen=True)
class _TreeNode:
    scope: tuple[int, ...]
    children: tuple[_TreeNode, ...] | None = None


@dataclass(frozen=True)
class _CircuitConfig:
    num_input_units: int
    num_sum_units: int
    input_layer: str
    num_states: int | None
    num_variables: int
    non_monotonic: bool
    non_monotonic_inputs: bool


def _validate_model_common(
    *,
    num_variables: int,
    num_input_units: int,
    num_sum_units: int,
    input_layer: str,
    image_shape: tuple[int, int, int] | None,
) -> None:
    if num_variables <= 0:
        raise InvalidParameterError(f"num_variables must be >= 1, got {num_variables}.")
    if num_input_units <= 0:
        raise InvalidParameterError(f"num_input_units must be >= 1, got {num_input_units}.")
    if num_sum_units <= 0:
        raise InvalidParameterError(f"num_sum_units must be >= 1, got {num_sum_units}.")

    if input_layer not in {"categorical", "embedding", "gaussian"}:
        raise InvalidParameterError(
            "input_layer must be one of {'categorical','embedding','gaussian'}, " f"got '{input_layer}'."
        )

    if image_shape is not None:
        if len(image_shape) != 3:
            raise ShapeError(f"image_shape must be (channels,height,width), got {image_shape}.")
        c, h, w = image_shape
        if c <= 0 or h <= 0 or w <= 0:
            raise ShapeError(f"image_shape entries must be > 0, got {image_shape}.")
        if c * h * w != num_variables:
            raise ShapeError(
                "image_shape product must match num_variables; "
                f"got {image_shape} -> {c*h*w} vs {num_variables}."
            )


def _resolve_num_states(input_layer: str, input_layer_kwargs: dict[str, int] | None) -> int | None:
    kwargs = {} if input_layer_kwargs is None else input_layer_kwargs
    if input_layer == "categorical":
        if "num_categories" not in kwargs:
            raise InvalidParameterError(
                "input_layer_kwargs['num_categories'] is required for categorical input."
            )
        return int(kwargs["num_categories"])
    if input_layer == "embedding":
        if "num_states" not in kwargs:
            raise InvalidParameterError("input_layer_kwargs['num_states'] is required for embedding input.")
        return int(kwargs["num_states"])
    return None


def _signed_init_bounds(num_variables: int) -> tuple[float, float]:
    if num_variables <= 2:
        return -2.0, 2.0
    return 0.0, 1.0


def _rand_uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> Tensor:
    t = torch.rand(shape, dtype=dtype, generator=generator)
    return (high - low) * t + low


def _make_leaf(
    scope: Iterable[int],
    cfg: _CircuitConfig,
    *,
    generator: torch.Generator,
) -> Module:
    scope_obj = Scope(list(scope))

    if cfg.input_layer == "gaussian":
        return Normal(scope=scope_obj, out_channels=cfg.num_input_units, num_repetitions=1)

    assert cfg.num_states is not None
    if cfg.input_layer in {"categorical", "embedding"} and cfg.non_monotonic and cfg.non_monotonic_inputs:
        low, high = _signed_init_bounds(cfg.num_variables)
        weights = _rand_uniform(
            (len(scope_obj.query), cfg.num_input_units, 1, cfg.num_states),
            low,
            high,
            generator=generator,
            dtype=torch.get_default_dtype(),
        )
        return SignedCategorical(
            scope=scope_obj,
            out_channels=cfg.num_input_units,
            num_repetitions=1,
            K=cfg.num_states,
            weights=weights,
        )

    # For categorical and monotone embedding, reuse positive categorical leaves.
    return Categorical(
        scope=scope_obj,
        out_channels=cfg.num_input_units,
        num_repetitions=1,
        K=cfg.num_states,
    )


def _make_signed_sum(
    inputs: Module,
    *,
    out_channels: int,
    cfg: _CircuitConfig,
    generator: torch.Generator,
) -> SignedSum:
    low, high = _signed_init_bounds(cfg.num_variables)
    weights = _rand_uniform(
        (
            inputs.out_shape.features,
            inputs.out_shape.channels,
            out_channels,
            inputs.out_shape.repetitions,
        ),
        low,
        high,
        generator=generator,
        dtype=torch.get_default_dtype(),
    )
    return SignedSum(
        inputs=inputs,
        out_channels=out_channels,
        num_repetitions=inputs.out_shape.repetitions,
        weights=weights,
    )


def _instantiate_tree(node: _TreeNode, cfg: _CircuitConfig, *, generator: torch.Generator) -> Module:
    if not node.children:
        return _make_leaf(node.scope, cfg, generator=generator)

    children = [_instantiate_tree(ch, cfg, generator=generator) for ch in node.children]
    prod = Product(children)
    if cfg.non_monotonic:
        return _make_signed_sum(prod, out_channels=cfg.num_sum_units, cfg=cfg, generator=generator)
    return Sum(inputs=prod, out_channels=cfg.num_sum_units, num_repetitions=prod.out_shape.repetitions)


def _finalize_scalar(
    root: Module,
    cfg: _CircuitConfig,
    *,
    generator: torch.Generator,
    force_root_sum: bool = False,
) -> Module:
    out = root
    # Ensure scalar feature output.
    if out.out_shape.features != 1:
        out = Product(out)

    if out.out_shape.channels != 1 or force_root_sum:
        if cfg.non_monotonic:
            out = _make_signed_sum(out, out_channels=1, cfg=cfg, generator=generator)
        else:
            out = Sum(inputs=out, out_channels=1, num_repetitions=out.out_shape.repetitions)

    if tuple(out.out_shape) != (1, 1, 1):
        raise ShapeError(f"Expected scalar out_shape (1,1,1), got {tuple(out.out_shape)}.")
    return out


def _build_component(
    tree: _TreeNode,
    cfg: _CircuitConfig,
    *,
    seed: int,
    force_root_sum: bool = False,
) -> Module:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    root = _instantiate_tree(tree, cfg, generator=gen)
    return _finalize_scalar(root, cfg, generator=gen, force_root_sum=force_root_sum)


def _split_random_binary(scope: list[int], *, rng: random.Random) -> tuple[list[int], list[int]]:
    perm = list(scope)
    rng.shuffle(perm)
    mid = len(perm) // 2
    return perm[:mid], perm[mid:]


def _build_random_binary_tree(scope: list[int], *, rng: random.Random) -> _TreeNode:
    if len(scope) <= 1:
        return _TreeNode(scope=tuple(scope), children=None)
    left, right = _split_random_binary(scope, rng=rng)
    if not left or not right:
        return _TreeNode(scope=tuple(scope), children=None)
    return _TreeNode(
        scope=tuple(scope),
        children=(
            _build_random_binary_tree(left, rng=rng),
            _build_random_binary_tree(right, rng=rng),
        ),
    )


def _build_linear_tree_from_order(order: list[int]) -> _TreeNode:
    if len(order) <= 1:
        return _TreeNode(scope=tuple(order), children=None)
    left = [order[0]]
    right = order[1:]
    return _TreeNode(
        scope=tuple(order),
        children=(
            _TreeNode(scope=tuple(left), children=None),
            _build_linear_tree_from_order(right),
        ),
    )


def _pixel_indices_for_patch(
    image_shape: tuple[int, int, int],
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> list[int]:
    c, h, w = image_shape
    _ = h
    out: list[int] = []
    for ch in range(c):
        base = ch * (image_shape[1] * image_shape[2])
        for y in range(y0, y1):
            for x in range(x0, x1):
                out.append(base + y * w + x)
    return out


def _build_quadtree_patch(
    image_shape: tuple[int, int, int],
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    num_patch_splits: int,
) -> _TreeNode:
    scope = _pixel_indices_for_patch(image_shape, y0=y0, y1=y1, x0=x0, x1=x1)
    hh = y1 - y0
    ww = x1 - x0
    if hh <= 1 and ww <= 1:
        return _TreeNode(scope=tuple(scope), children=None)

    children_bounds: list[tuple[int, int, int, int]] = []
    if num_patch_splits == 2:
        if hh >= ww and hh > 1:
            ym = y0 + hh // 2
            children_bounds = [(y0, ym, x0, x1), (ym, y1, x0, x1)]
        elif ww > 1:
            xm = x0 + ww // 2
            children_bounds = [(y0, y1, x0, xm), (y0, y1, xm, x1)]
    else:
        ym = y0 + hh // 2
        xm = x0 + ww // 2
        children_bounds = [
            (y0, ym, x0, xm),
            (y0, ym, xm, x1),
            (ym, y1, x0, xm),
            (ym, y1, xm, x1),
        ]

    child_nodes: list[_TreeNode] = []
    for cy0, cy1, cx0, cx1 in children_bounds:
        if cy1 <= cy0 or cx1 <= cx0:
            continue
        child_nodes.append(
            _build_quadtree_patch(
                image_shape,
                y0=cy0,
                y1=cy1,
                x0=cx0,
                x1=cx1,
                num_patch_splits=num_patch_splits,
            )
        )

    if len(child_nodes) < 2:
        return _TreeNode(scope=tuple(scope), children=None)
    return _TreeNode(scope=tuple(scope), children=tuple(child_nodes))


def _build_tree(
    *,
    region_graph: str,
    num_variables: int,
    image_shape: tuple[int, int, int] | None,
    seed: int,
) -> _TreeNode:
    rng = random.Random(int(seed))

    if region_graph == "rnd-bt":
        return _build_random_binary_tree(list(range(num_variables)), rng=rng)

    if region_graph == "rnd-lt":
        order = list(range(num_variables))
        rng.shuffle(order)
        return _build_linear_tree_from_order(order)

    if region_graph == "lt":
        return _build_linear_tree_from_order(list(range(num_variables)))

    if region_graph in {"qt", "qt-4", "qt-2"}:
        if image_shape is None:
            raise InvalidParameterError(
                f"region_graph='{region_graph}' requires image_shape=(channels,height,width)."
            )
        _, h, w = image_shape
        splits = 2 if region_graph == "qt-2" else 4
        return _build_quadtree_patch(image_shape, y0=0, y1=h, x0=0, x1=w, num_patch_splits=splits)

    raise InvalidParameterError(
        "Unsupported region_graph. Expected one of {'rnd-bt','rnd-lt','lt','qt','qt-2','qt-4'}, "
        f"got '{region_graph}'."
    )


def _structure_seed(base_seed: int, index: int, structured_decomposable: bool) -> int:
    if structured_decomposable:
        return int(base_seed)
    return int(base_seed + index * 123)


class SOSModel(Module):
    """High-level SOS constructor with reference-style arguments.

    The model builds ``num_squares`` non-monotonic components and wraps them in
    :class:`spflow.zoo.sos.SOCS`. When ``complex=True``, each square is represented
    by two real components (real/imag), yielding equivalent ``|c(x)|^2`` semantics.
    """

    def __init__(
        self,
        num_variables: int,
        image_shape: tuple[int, int, int] | None = None,
        *,
        num_input_units: int,
        num_sum_units: int,
        input_layer: str,
        input_layer_kwargs: dict[str, int] | None = None,
        num_squares: int = 1,
        region_graph: str = "rnd-bt",
        structured_decomposable: bool = False,
        complex: bool = False,
        non_monotonic_inputs: bool | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if num_squares <= 0:
            raise InvalidParameterError(f"num_squares must be >= 1, got {num_squares}.")

        _validate_model_common(
            num_variables=num_variables,
            num_input_units=num_input_units,
            num_sum_units=num_sum_units,
            input_layer=input_layer,
            image_shape=image_shape,
        )

        if non_monotonic_inputs is None:
            non_monotonic_inputs = input_layer == "embedding"

        num_states = _resolve_num_states(input_layer, input_layer_kwargs)
        cfg = _CircuitConfig(
            num_input_units=num_input_units,
            num_sum_units=num_sum_units,
            input_layer=input_layer,
            num_states=num_states,
            num_variables=num_variables,
            non_monotonic=True,
            non_monotonic_inputs=bool(non_monotonic_inputs),
        )

        comps: list[Module] = []
        complex_pairs: list[tuple[Module, Module]] = []
        for i in range(num_squares):
            s_seed = _structure_seed(seed, i, structured_decomposable)
            tree = _build_tree(
                region_graph=region_graph,
                num_variables=num_variables,
                image_shape=image_shape,
                seed=s_seed,
            )

            if complex:
                real = _build_component(tree, cfg, seed=seed + 10_000 + 2 * i)
                imag = _build_component(tree, cfg, seed=seed + 10_000 + 2 * i + 1)
                complex_pairs.append((real, imag))
                comps.extend([real, imag])
            else:
                comps.append(_build_component(tree, cfg, seed=seed + 10_000 + i))

        self._complex_component_pairs = complex_pairs
        self.socs = SOCS(comps)

        self.scope = self.socs.scope
        self.in_shape = self.socs.in_shape
        self.out_shape = self.socs.out_shape

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self.socs.feature_to_scope

    @property
    def components(self) -> list[Module]:
        return [cast(Module, c) for c in self.socs.components]

    @property
    def complex_component_pairs(self) -> list[tuple[Module, Module]]:
        return list(self._complex_component_pairs)

    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:
        return self.socs.log_likelihood(data, cache=cache)

    def _expectation_maximization_step(
        self,
        data: Tensor,
        bias_correction: bool = True,
        *,
        cache: Cache,
    ) -> None:
        self.socs._expectation_maximization_step(data, bias_correction=bias_correction, cache=cache)

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        return self.socs.marginalize(marg_rvs, prune=prune, cache=cache)

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
    ) -> Tensor:
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full((num_samples, len(self.scope.query)), float("nan"), device=self.device)
        if cache is None:
            cache = Cache()
        sampling_ctx = build_root_sampling_context(
            num_samples=data.shape[0],
            num_features=self.socs.out_shape.features,
            device=data.device,
        )
        return self.socs._sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=sampling_ctx)

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> Tensor:
        return self.socs._sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=sampling_ctx)


class ExpSOSModel(Module):
    """High-level ExpSOS/µSOCS constructor with reference-style arguments."""

    def __init__(
        self,
        num_variables: int,
        image_shape: tuple[int, int, int] | None = None,
        *,
        num_input_units: int,
        num_sum_units: int,
        mono_num_input_units: int = 2,
        mono_num_sum_units: int = 2,
        input_layer: str,
        input_layer_kwargs: dict[str, int] | None = None,
        region_graph: str = "rnd-bt",
        structured_decomposable: bool = False,
        complex: bool = False,
        non_monotonic_inputs: bool | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        _validate_model_common(
            num_variables=num_variables,
            num_input_units=num_input_units,
            num_sum_units=num_sum_units,
            input_layer=input_layer,
            image_shape=image_shape,
        )

        if mono_num_input_units <= 0:
            raise InvalidParameterError(f"mono_num_input_units must be >= 1, got {mono_num_input_units}.")
        if mono_num_sum_units <= 0:
            raise InvalidParameterError(f"mono_num_sum_units must be >= 1, got {mono_num_sum_units}.")

        if non_monotonic_inputs is None:
            non_monotonic_inputs = input_layer == "embedding"

        num_states = _resolve_num_states(input_layer, input_layer_kwargs)

        comp_cfg = _CircuitConfig(
            num_input_units=num_input_units,
            num_sum_units=num_sum_units,
            input_layer=input_layer,
            num_states=num_states,
            num_variables=num_variables,
            non_monotonic=True,
            non_monotonic_inputs=bool(non_monotonic_inputs),
        )

        mono_cfg = _CircuitConfig(
            num_input_units=mono_num_input_units,
            num_sum_units=mono_num_sum_units,
            input_layer=input_layer,
            num_states=num_states,
            num_variables=num_variables,
            non_monotonic=False,
            non_monotonic_inputs=False,
        )

        # ExpSOS uses one region graph for both monotone and non-monotone circuits.
        s_seed = _structure_seed(seed, 0, structured_decomposable)
        tree = _build_tree(
            region_graph=region_graph,
            num_variables=num_variables,
            image_shape=image_shape,
            seed=s_seed,
        )

        monotone = _build_component(
            tree,
            mono_cfg,
            seed=seed + 20_000,
            force_root_sum=True,
        )

        if complex:
            real = _build_component(tree, comp_cfg, seed=seed + 21_000, force_root_sum=True)
            imag = _build_component(tree, comp_cfg, seed=seed + 21_001, force_root_sum=True)
            components = [real, imag]
            self._complex_component_pairs = [(real, imag)]
        else:
            single = _build_component(tree, comp_cfg, seed=seed + 21_000, force_root_sum=True)
            components = [single]
            self._complex_component_pairs = []

        self.monotone = monotone
        self._components = components
        self.exp_socs = ExpSOCS(monotone=monotone, components=components)

        self.scope = self.exp_socs.scope
        self.in_shape = self.exp_socs.in_shape
        self.out_shape = self.exp_socs.out_shape

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self.exp_socs.feature_to_scope

    @property
    def components(self) -> list[Module]:
        return list(self._components)

    @property
    def complex_component_pairs(self) -> list[tuple[Module, Module]]:
        return list(self._complex_component_pairs)

    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:
        return self.exp_socs.log_likelihood(data, cache=cache)

    def _expectation_maximization_step(
        self,
        data: Tensor,
        bias_correction: bool = True,
        *,
        cache: Cache,
    ) -> None:
        self.exp_socs._expectation_maximization_step(data, bias_correction=bias_correction, cache=cache)

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        return self.exp_socs.marginalize(marg_rvs, prune=prune, cache=cache)

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
    ) -> Tensor:
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full((num_samples, len(self.scope.query)), float("nan"), device=self.device)
        if cache is None:
            cache = Cache()
        sampling_ctx = build_root_sampling_context(
            num_samples=data.shape[0],
            num_features=self.exp_socs.out_shape.features,
            device=data.device,
        )
        return self.exp_socs._sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=sampling_ctx)

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> Tensor:
        return self.exp_socs._sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=sampling_ctx)
