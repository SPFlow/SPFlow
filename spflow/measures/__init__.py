"""Measure utilities for SPFlow probabilistic circuits.

This package provides information-theoretic measures and explainability utilities
that operate on SPFlow :class:`spflow.modules.module.Module` objects.
"""

from spflow.measures.information_theory import (
    conditional_mutual_information,
    entropy,
    mutual_information,
)
from spflow.measures.weight_of_evidence import (
    conditional_probability,
    weight_of_evidence,
    weight_of_evidence_leave_one_out,
)

__all__ = [
    "entropy",
    "mutual_information",
    "conditional_mutual_information",
    "conditional_probability",
    "weight_of_evidence",
    "weight_of_evidence_leave_one_out",
]
