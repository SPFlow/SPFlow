"""Structural compatibility checks for sets of probabilistic circuits.

SOCS (Σ2cmp) assumes that its component circuits are compatible / share the same
structured decomposition. In SPFlow, we approximate this by checking that a set
of modules has the same "skeleton":

- Same scope at each corresponding node
- Same module types and arities
- Same Cat concatenation dimension (`dim`)
- Same CLTree structure (parents, K) where relevant

These checks are conservative: they may reject some circuits that are compatible
in a more general theoretical sense, but they provide a practical safeguard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from spflow.exceptions import ShapeError, StructureError
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.cltree import CLTree
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module
from spflow.modules.ops.cat import Cat
from spflow.modules.products.product import Product
from spflow.modules.sos.socs import SOCS
from spflow.modules.sums.signed_sum import SignedSum
from spflow.modules.sums.sum import Sum


@dataclass(frozen=True)
class CompatibilityIssue:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def _scope_equal(a: Scope, b: Scope) -> bool:
    return a == b


def _children(module: Module) -> list[Module]:
    if isinstance(module, Cat):
        return [cast(Module, m) for m in module.inputs]
    if isinstance(module, (Sum, SignedSum)):
        return [cast(Module, module.inputs)]
    if isinstance(module, Product):
        return [cast(Module, module.inputs)]
    # Leaf or unknown node types have no structural children in this checker.
    return []


def _check_pair(a: Module, b: Module, *, path: str) -> None:
    if type(a) is not type(b):
        raise StructureError(f"{path}: type mismatch: {type(a).__name__} vs {type(b).__name__}.")

    if not _scope_equal(a.scope, b.scope):
        raise ShapeError(f"{path}: scope mismatch: {a.scope} vs {b.scope}.")

    if tuple(a.out_shape) != tuple(b.out_shape):
        raise ShapeError(f"{path}: out_shape mismatch: {tuple(a.out_shape)} vs {tuple(b.out_shape)}.")

    if isinstance(a, Cat):
        if a.dim != cast(Cat, b).dim:
            raise StructureError(f"{path}: Cat dim mismatch: {a.dim} vs {cast(Cat, b).dim}.")
        if len(a.inputs) != len(cast(Cat, b).inputs):
            raise StructureError(
                f"{path}: Cat arity mismatch: {len(a.inputs)} vs {len(cast(Cat, b).inputs)}."
            )

    if isinstance(a, Categorical):
        if a.K != cast(Categorical, b).K:
            raise ShapeError(f"{path}: Categorical K mismatch: {a.K} vs {cast(Categorical, b).K}.")

    if isinstance(a, CLTree):
        bb = cast(CLTree, b)
        if a.K != bb.K:
            raise ShapeError(f"{path}: CLTree K mismatch: {a.K} vs {bb.K}.")
        if not torch.equal(a.parents, bb.parents):
            raise StructureError(f"{path}: CLTree parents mismatch (structure differs).")

    # Recurse
    ca = _children(a)
    cb = _children(b)
    if len(ca) != len(cb):
        raise StructureError(f"{path}: child count mismatch: {len(ca)} vs {len(cb)}.")
    for i, (ai, bi) in enumerate(zip(ca, cb)):
        _check_pair(ai, bi, path=f"{path}.inputs[{i}]")


def check_compatible_components(components: list[Module]) -> None:
    """Raise if components are not structurally compatible."""
    if len(components) < 2:
        return

    ref = components[0]
    for i, other in enumerate(components[1:], start=1):
        _check_pair(ref, other, path=f"components[0] vs components[{i}]")


def check_socs_compatibility(model: SOCS) -> None:
    """Convenience wrapper for SOCS."""
    check_compatible_components([cast(Module, m) for m in model.components])
