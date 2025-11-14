"""Metadata and data handling for probabilistic circuits.

This module provides classes and utilities for handling metadata related to
probabilistic circuits, including scope management, feature contexts, and
type systems for structured probabilistic modeling. These metadata components are essential for ensuring valid circuit construction
and proper variable handling throughout the learning and inference processes.
"""

from . import data
from .data import Scope, FeatureContext, FeatureTypes
