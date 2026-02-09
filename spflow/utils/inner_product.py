"""Exact inner-product utilities for probabilistic circuits.

This module is the canonical public entry point for exact inner products used in
SPFlow's SOS/SOCS-style normalization routines.

The implementation lives in `spflow/utils/inner_product_core.py` and is shared
with `spflow/zoo/sos/inner_product.py` to avoid duplicated math.
"""

from __future__ import annotations

from torch import Tensor

from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module
from spflow.modules.sums.signed_sum import SignedSum
from spflow.utils.cache import Cache
from spflow.utils.inner_product_core import (
    inner_product_matrix as _inner_product_matrix,
    leaf_inner_product,
    log_self_inner_product_scalar as _log_self_inner_product_scalar,
    triple_product_scalar as _triple_product_scalar,
    triple_product_tensor as _triple_product_tensor,
)


def _prepare_legacy_memo_alias(cache: Cache | None, primary: str, legacy: str) -> None:
    if cache is None:
        return
    if primary not in cache.extras and legacy in cache.extras:
        cache.extras[primary] = cache.extras[legacy]


def _sync_legacy_memo_alias(cache: Cache | None, primary: str, legacy: str) -> None:
    if cache is None:
        return
    memo = cache.extras.get(primary)
    if isinstance(memo, dict):
        cache.extras[legacy] = memo


def inner_product_matrix(a: Module, b: Module, *, cache: Cache | None = None) -> Tensor:
    primary_key = "_inner_product_memo"
    legacy_key = "_sos_inner_product_memo"
    _prepare_legacy_memo_alias(cache, primary_key, legacy_key)
    out = _inner_product_matrix(
        a,
        b,
        cache=cache,
        signed_sum_types=(SignedSum,),
        memo_key=primary_key,
    )
    _sync_legacy_memo_alias(cache, primary_key, legacy_key)
    return out


def log_self_inner_product_scalar(module: Module, *, cache: Cache | None = None) -> Tensor:
    primary_key = "_inner_product_memo"
    legacy_key = "_sos_inner_product_memo"
    _prepare_legacy_memo_alias(cache, primary_key, legacy_key)
    out = _log_self_inner_product_scalar(
        module,
        cache=cache,
        signed_sum_types=(SignedSum,),
        memo_key=primary_key,
    )
    _sync_legacy_memo_alias(cache, primary_key, legacy_key)
    return out


def triple_product_tensor(a: Module, b: Module, c: Module, *, cache: Cache | None = None) -> Tensor:
    primary_key = "_inner_product_memo"
    legacy_key = "_sos_triple_product_memo"
    _prepare_legacy_memo_alias(cache, primary_key, legacy_key)
    out = _triple_product_tensor(
        a,
        b,
        c,
        cache=cache,
        signed_sum_types=(SignedSum,),
        memo_key=primary_key,
    )
    _sync_legacy_memo_alias(cache, primary_key, legacy_key)
    return out


def triple_product_scalar(a: Module, b: Module, c: Module, *, cache: Cache | None = None) -> Tensor:
    primary_key = "_inner_product_memo"
    legacy_key = "_sos_triple_product_memo"
    _prepare_legacy_memo_alias(cache, primary_key, legacy_key)
    out = _triple_product_scalar(
        a,
        b,
        c,
        cache=cache,
        signed_sum_types=(SignedSum,),
        memo_key=primary_key,
    )
    _sync_legacy_memo_alias(cache, primary_key, legacy_key)
    return out


__all__ = [
    "LeafModule",
    "leaf_inner_product",
    "inner_product_matrix",
    "log_self_inner_product_scalar",
    "triple_product_tensor",
    "triple_product_scalar",
]
