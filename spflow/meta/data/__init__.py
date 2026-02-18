"""Data structures for metadata management in probabilistic circuits.

This module provides core data structures for managing metadata, including
variable scopes and interval evidence needed for proper circuit construction
and validation. These structures ensure type safety and proper variable
handling throughout the SPFlow ecosystem.
"""

from .interval_evidence import IntervalEvidence
from .scope import Scope
