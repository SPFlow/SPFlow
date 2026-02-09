"""Exact inner-/triple-product utilities for SOS/SOCS circuits (zoo version).

This module wraps `spflow.utils.inner_product_core` but uses the zoo-specific
`SignedSum` type and separate cache memo namespaces.
"""

from __future__ import annotations

from torch import Tensor

from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module
from spflow.utils.cache import Cache
from spflow.utils.inner_product_core import (
    inner_product_matrix as _inner_product_matrix,
    leaf_inner_product,
    log_self_inner_product_scalar as _log_self_inner_product_scalar,
    triple_product_scalar as _triple_product_scalar,
    triple_product_tensor as _triple_product_tensor,
)
from spflow.modules.sums.signed_sum import SignedSum


def inner_product_matrix(a: Module, b: Module, *, cache: Cache | None = None) -> Tensor:
    return _inner_product_matrix(
        a,
        b,
        cache=cache,
        signed_sum_types=(SignedSum,),
        memo_key="_sos_inner_product_memo",
    )


def log_self_inner_product_scalar(module: Module, *, cache: Cache | None = None) -> Tensor:
    return _log_self_inner_product_scalar(
        module,
        cache=cache,
        signed_sum_types=(SignedSum,),
        memo_key="_sos_inner_product_memo",
    )


def triple_product_tensor(a: Module, b: Module, c: Module, *, cache: Cache | None = None) -> Tensor:
    return _triple_product_tensor(
        a,
        b,
        c,
        cache=cache,
        signed_sum_types=(SignedSum,),
        memo_key="_sos_triple_product_memo",
    )


def triple_product_scalar(a: Module, b: Module, c: Module, *, cache: Cache | None = None) -> Tensor:
    return _triple_product_scalar(
        a,
        b,
        c,
        cache=cache,
        signed_sum_types=(SignedSum,),
        memo_key="_sos_triple_product_memo",
    )


__all__ = [
    "LeafModule",
    "leaf_inner_product",
    "inner_product_matrix",
    "log_self_inner_product_scalar",
    "triple_product_tensor",
    "triple_product_scalar",
]
