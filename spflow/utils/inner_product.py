"""Exact inner-product utilities for probabilistic circuits.

This module is the canonical public entry point for exact inner products used in
SPFlow's SOS/SOCS-style normalization routines.

The implementation lives in `spflow/utils/inner_product_core.py` and is shared
with `spflow/zoo/sos/inner_product.py` to avoid duplicated math.
"""

from __future__ import annotations

from torch import Tensor

from spflow.modules.module import Module
from spflow.modules.sums.signed_sum import SignedSum
from spflow.utils.cache import Cache
from spflow.utils.inner_product_core import (
    inner_product_matrix as _inner_product_matrix,
    leaf_inner_product,
    log_self_inner_product_scalar as _log_self_inner_product_scalar,
)


def inner_product_matrix(a: Module, b: Module, *, cache: Cache | None = None) -> Tensor:
    return _inner_product_matrix(
        a,
        b,
        cache=cache,
        signed_sum_types=(SignedSum,),
        memo_key="_inner_product_memo",
    )


def log_self_inner_product_scalar(module: Module, *, cache: Cache | None = None) -> Tensor:
    return _log_self_inner_product_scalar(
        module,
        cache=cache,
        signed_sum_types=(SignedSum,),
        memo_key="_inner_product_memo",
    )

