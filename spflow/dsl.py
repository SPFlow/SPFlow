"""Construction-only DSL for building SPFlow circuits.

This module provides a small, non-invasive expression layer for writing examples
with algebraic syntax while keeping the core `spflow.modules` API unchanged.

The DSL is intentionally minimal:

- Products: `term(A) * term(B)`
- Weighted sums (mixtures): `0.4 * term(A) + 0.6 * term(B)`

To obtain an actual `Module`, call `.build()` on the resulting expression.

Notes:
    - Weights must be provided for sums; `term(A) + term(B)` is intentionally disallowed.
    - Weighted sums are restricted to terms with `out_shape.channels == 1` for simplicity.

For convenience in docs/examples, `dsl()` can temporarily enable operator overloads on
`spflow.modules.module.Module` within a context manager, so that expressions like
`0.4 * Normal(0) * Normal(1) + 0.6 * Normal(0) * Normal(1)` work without wrapping leaves.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from einops import repeat

from spflow.exceptions import (
    InvalidParameterCombinationError,
    InvalidParameterError,
    InvalidWeightsError,
    ScopeError,
    ShapeError,
)
from spflow.meta.data.scope import Scope
from spflow.modules.module import Module
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum


@runtime_checkable
class Buildable(Protocol):
    """Protocol for objects that can build a `Module`."""

    def build(self) -> Module:
        ...


def as_expr(value: Module | Buildable) -> Buildable:
    """Convert a `Module` or DSL expression to a DSL expression."""
    if isinstance(value, Module):
        return Term(value)
    if isinstance(value, Buildable):
        return value
    raise InvalidParameterError(f"Expected a Module or DSL expression, got {type(value)}.")


def term(module: Module) -> "Term":
    """Wrap a `Module` as a DSL term."""
    return Term(module)


def w(weight: float, value: Module | Buildable) -> "WeightedExpr":
    """Convenience helper to create a weighted term."""
    return WeightedExpr(weight=weight, expr=as_expr(value))


def build(value: Module | Buildable) -> Module:
    """Build a concrete `Module` from a DSL expression (or pass through `Module`)."""
    if isinstance(value, Module):
        return value
    return value.build()


@dataclass(frozen=True)
class Term(Buildable):
    """Leaf expression node that wraps a concrete `Module`."""

    module: Module

    def build(self) -> Module:
        return self.module

    def __mul__(self, other: object) -> "ProductExpr | WeightedExpr":
        if isinstance(other, (int, float)):
            return WeightedExpr(weight=float(other), expr=self)
        return ProductExpr([self, as_expr(other)])  # type: ignore[arg-type]

    def __rmul__(self, weight: float) -> "WeightedExpr":
        if not isinstance(weight, (int, float)):
            raise InvalidParameterError(f"Expected numeric weight, got {type(weight)}.")
        return WeightedExpr(weight=float(weight), expr=self)

    def __add__(self, other: object) -> "SumExpr":  # pragma: no cover
        raise InvalidParameterError(
            "Unweighted '+' is not supported in the DSL. Use 'a * term(x) + b * term(y)'."
        )


@dataclass(frozen=True)
class ProductExpr(Buildable):
    """Product of one or more sub-expressions."""

    factors: list[Buildable]

    def __mul__(self, other: object) -> "ProductExpr | WeightedExpr":
        if isinstance(other, (int, float)):
            return WeightedExpr(weight=float(other), expr=self)
        return ProductExpr([*self.factors, as_expr(other)])  # type: ignore[arg-type]

    def __rmul__(self, weight: float) -> "WeightedExpr":
        if not isinstance(weight, (int, float)):
            raise InvalidParameterError(f"Expected numeric weight, got {type(weight)}.")
        return WeightedExpr(weight=float(weight), expr=self)

    def __add__(self, other: object) -> "SumExpr":  # pragma: no cover
        raise InvalidParameterError(
            "Unweighted '+' is not supported in the DSL. Use 'a * term(x) + b * term(y)'."
        )

    def build(self) -> Module:
        modules = [factor.build() for factor in self.factors]
        _validate_product_modules(modules)
        return Product(inputs=modules)


@dataclass(frozen=True)
class WeightedExpr:
    """A weighted expression term used as an input to mixtures."""

    weight: float
    expr: Buildable

    def __post_init__(self) -> None:
        if not isinstance(self.weight, (int, float)):
            raise InvalidParameterError(f"Weight must be numeric, got {type(self.weight)}.")
        if not torch.isfinite(torch.as_tensor(float(self.weight))):
            raise InvalidWeightsError("Weight must be finite.")
        if float(self.weight) <= 0.0:
            raise InvalidWeightsError("Weights must be strictly positive.")

    def build(self) -> Module:
        raise InvalidParameterError(
            "A weighted term cannot be built directly. Combine weighted terms with '+' to form a mixture."
        )

    def __add__(self, other: "WeightedExpr | SumExpr") -> "SumExpr":
        if isinstance(other, WeightedExpr):
            return SumExpr([(self.weight, self.expr), (other.weight, other.expr)])
        if isinstance(other, SumExpr):
            return SumExpr([(self.weight, self.expr), *other.terms])
        raise InvalidParameterError(
            f"Can only add a weighted term to another weighted term or mixture, got {type(other)}."
        )

    def __radd__(self, other: object) -> "SumExpr":
        if isinstance(other, SumExpr):
            return SumExpr([*other.terms, (self.weight, self.expr)])
        return NotImplemented  # type: ignore[return-value]

    def __mul__(self, other: Module | Buildable) -> "WeightedExpr":
        """Apply this weight to a product expression.

        This enables compact example syntax like ``0.4 * Normal(0) * Normal(1)``.
        """
        return WeightedExpr(weight=self.weight, expr=ProductExpr([self.expr, as_expr(other)]))


@dataclass(frozen=True)
class SumExpr(Buildable):
    """A weighted mixture of expressions.

    Terms are stored as (weight, expr) pairs and normalized on build.
    """

    terms: list[tuple[float, Buildable]]

    def __post_init__(self) -> None:
        if len(self.terms) < 2:
            raise InvalidParameterError("A mixture requires at least two weighted terms.")
        for weight, _ in self.terms:
            if float(weight) <= 0.0:
                raise InvalidWeightsError("Weights must be strictly positive.")

    def __add__(self, other: WeightedExpr | "SumExpr") -> "SumExpr":
        if isinstance(other, WeightedExpr):
            return SumExpr([*self.terms, (other.weight, other.expr)])
        if isinstance(other, SumExpr):
            return SumExpr([*self.terms, *other.terms])
        raise InvalidParameterError(
            f"Can only add a weighted term or mixture to a mixture, got {type(other)}."
        )

    def __mul__(self, other: object) -> ProductExpr | WeightedExpr:
        if isinstance(other, (int, float)):
            return WeightedExpr(weight=float(other), expr=self)
        return ProductExpr([self, as_expr(other)])  # type: ignore[arg-type]

    def __rmul__(self, weight: float) -> WeightedExpr:
        if not isinstance(weight, (int, float)):
            raise InvalidParameterError(f"Expected numeric weight, got {type(weight)}.")
        return WeightedExpr(weight=float(weight), expr=self)

    def build(self) -> Module:
        modules = [expr.build() for _, expr in self.terms]
        weights = [float(w) for w, _ in self.terms]

        _validate_sum_modules(modules)
        weights_tensor = _make_sum_weights(
            weights=weights,
            features=modules[0].out_shape.features,
            repetitions=modules[0].out_shape.repetitions,
            device=modules[0].device,
            dtype=torch.get_default_dtype(),
        )

        return Sum(inputs=modules, weights=weights_tensor)


def _validate_product_modules(modules: list[Module]) -> None:
    if len(modules) < 2:
        raise InvalidParameterError("Product requires at least two factors.")

    scopes = [m.scope for m in modules]
    if not Scope.all_pairwise_disjoint(scopes):
        raise ScopeError("Product factors must have disjoint scopes.")

    channels = {m.out_shape.channels for m in modules}
    if len(channels) != 1:
        raise ShapeError(f"Product factors must have the same out_channels; got {sorted(channels)}.")

    repetitions = {m.out_shape.repetitions for m in modules}
    if len(repetitions) != 1:
        raise ShapeError(f"Product factors must have the same num_repetitions; got {sorted(repetitions)}.")

    devices = {str(m.device) for m in modules}
    if len(devices) != 1:
        raise InvalidParameterCombinationError(
            f"Product factors must be on the same device; got {sorted(devices)}."
        )


def _validate_sum_modules(modules: list[Module]) -> None:
    if len(modules) < 2:
        raise InvalidParameterError("Sum requires at least two terms.")

    scopes = [m.scope for m in modules]
    if not Scope.all_equal(scopes):
        raise ScopeError("Sum terms must have identical scopes.")

    features = {m.out_shape.features for m in modules}
    if len(features) != 1:
        raise ShapeError(f"Sum terms must have the same number of features; got {sorted(features)}.")

    channels = {m.out_shape.channels for m in modules}
    if channels != {1}:
        raise ShapeError(
            "Sum DSL only supports terms with out_shape.channels == 1. "
            f"Got out_channels: {sorted(channels)}."
        )

    repetitions = {m.out_shape.repetitions for m in modules}
    if len(repetitions) != 1:
        raise ShapeError(f"Sum terms must have the same num_repetitions; got {sorted(repetitions)}.")

    devices = {str(m.device) for m in modules}
    if len(devices) != 1:
        raise InvalidParameterCombinationError(
            f"Sum terms must be on the same device; got {sorted(devices)}."
        )


def _make_sum_weights(
    *,
    weights: list[float],
    features: int,
    repetitions: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    in_channels = len(weights)
    raw = torch.as_tensor(weights, dtype=dtype, device=device)
    if raw.dim() != 1 or raw.shape[0] != in_channels:
        raise ShapeError("Expected a 1D weight vector.")
    if not torch.isfinite(raw).all():
        raise InvalidWeightsError("Weights must be finite.")
    if not torch.all(raw > 0):
        raise InvalidWeightsError("Weights must be strictly positive.")

    total = torch.sum(raw)
    if not torch.isfinite(total) or float(total) <= 0.0:
        raise InvalidWeightsError("Sum of weights must be finite and > 0.")

    normalized = raw / total

    # Sum expects weights of shape: (features, in_channels, out_channels, repetitions)
    w = repeat(normalized, "ci -> f ci 1 r", f=features, r=repetitions)
    return w


@contextmanager
def dsl():
    """Temporarily enable DSL operator overloads on `Module`.

    This is intended for documentation/examples. It monkeypatches operator methods on
    `spflow.modules.module.Module` for the duration of the context manager and restores
    the original methods afterward.

    Within the context:
    - `Module * Module` builds a `ProductExpr`
    - `float * Module` and `Module * float` create a `WeightedExpr`
    - `WeightedExpr + WeightedExpr (+ ...)` creates a `SumExpr`
    - `Module + Module` remains disallowed (weights must be explicit)
    """

    # Save originals (may be missing on base class).
    sentinel = object()
    orig_mul = getattr(Module, "__mul__", sentinel)
    orig_rmul = getattr(Module, "__rmul__", sentinel)
    orig_add = getattr(Module, "__add__", sentinel)
    orig_radd = getattr(Module, "__radd__", sentinel)

    def _dsl_mul(self: Module, other: object):
        if isinstance(other, (int, float)):
            return WeightedExpr(weight=float(other), expr=Term(self))
        if isinstance(other, WeightedExpr):
            return WeightedExpr(weight=other.weight, expr=ProductExpr([Term(self), other.expr]))
        if isinstance(other, Module):
            return ProductExpr([Term(self), Term(other)])
        if isinstance(other, Buildable):
            return ProductExpr([Term(self), other])
        return NotImplemented

    def _dsl_rmul(self: Module, other: object):
        if isinstance(other, (int, float)):
            return WeightedExpr(weight=float(other), expr=Term(self))
        return NotImplemented

    def _dsl_add(self: Module, other: object):
        raise InvalidParameterError("Unweighted '+' is not supported in the DSL. Use 'a * X + b * Y'.")

    def _dsl_radd(self: Module, other: object):
        return NotImplemented

    setattr(Module, "__mul__", _dsl_mul)
    setattr(Module, "__rmul__", _dsl_rmul)
    setattr(Module, "__add__", _dsl_add)
    setattr(Module, "__radd__", _dsl_radd)

    try:
        yield
    finally:
        # Restore originals.
        if orig_mul is sentinel:
            delattr(Module, "__mul__")
        else:
            setattr(Module, "__mul__", orig_mul)

        if orig_rmul is sentinel:
            delattr(Module, "__rmul__")
        else:
            setattr(Module, "__rmul__", orig_rmul)

        if orig_add is sentinel:
            delattr(Module, "__add__")
        else:
            setattr(Module, "__add__", orig_add)

        if orig_radd is sentinel:
            delattr(Module, "__radd__")
        else:
            setattr(Module, "__radd__", orig_radd)
