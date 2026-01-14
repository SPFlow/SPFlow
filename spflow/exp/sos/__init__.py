"""Experimental Sum-of-Squares circuit modules (SOCS).

This experimental module provides SOCS (Sum of Compatible Squares) implementation
for representing non-negative densities via squared signed circuits.

Warning:
    This module is experimental and may change in future versions.

Example::

    from spflow.exp.sos import SOCS, SignedSum, build_socs
    # Build from template or manually construct components
    model = build_socs(template, num_components=3, signed=True)
"""

from spflow.exp.sos.socs import SOCS
from spflow.exp.sos.signed_sum import SignedSum
from spflow.exp.sos.build_socs import build_socs, build_abs_weight_proposal
from spflow.exp.sos.compatibility import check_compatible_components, check_socs_compatibility
from spflow.exp.sos.inner_product import (
    inner_product_matrix,
    leaf_inner_product,
    log_self_inner_product_scalar,
)
from spflow.exp.sos.signed_semiring import signed_logsumexp, sign_of, logabs_of

__all__ = [
    "SOCS",
    "SignedSum",
    "build_socs",
    "build_abs_weight_proposal",
    "check_compatible_components",
    "check_socs_compatibility",
    "inner_product_matrix",
    "leaf_inner_product",
    "log_self_inner_product_scalar",
    "signed_logsumexp",
    "sign_of",
    "logabs_of",
]
