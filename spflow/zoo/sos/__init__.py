"""Paper Zoo: Sum-of-Squares circuit modules (SOCS).

This module provides SOCS (Sum of Compatible Squares) implementation
for representing non-negative densities via squared signed circuits.

Example::

    from spflow.zoo.sos import SOCS, SignedSum, build_socs
    # Build from template or manually construct components
    model = build_socs(template, num_components=3, signed=True)
"""

from spflow.modules.sos.socs import SOCS
from spflow.modules.sums.signed_sum import SignedSum
from spflow.zoo.sos.signed_categorical import SignedCategorical
from spflow.learn.build_socs import build_socs, build_abs_weight_proposal, build_complex_socs
from spflow.zoo.sos.exp_socs import ExpSOCS
from spflow.zoo.sos.models import SOSModel, ExpSOSModel
from spflow.utils.compatibility import check_compatible_components, check_socs_compatibility
from spflow.utils.inner_product import (
    inner_product_matrix,
    leaf_inner_product,
    log_self_inner_product_scalar,
    triple_product_scalar,
)
from spflow.utils.signed_semiring import signed_logsumexp, sign_of, logabs_of

__all__ = [
    "SOCS",
    "SignedSum",
    "SignedCategorical",
    "build_socs",
    "build_abs_weight_proposal",
    "build_complex_socs",
    "ExpSOCS",
    "SOSModel",
    "ExpSOSModel",
    "check_compatible_components",
    "check_socs_compatibility",
    "inner_product_matrix",
    "leaf_inner_product",
    "log_self_inner_product_scalar",
    "triple_product_scalar",
    "signed_logsumexp",
    "sign_of",
    "logabs_of",
]
