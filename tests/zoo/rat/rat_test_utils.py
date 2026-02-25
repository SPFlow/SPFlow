"""Shared RAT-SPN test utilities."""

from __future__ import annotations

from spflow.meta import Scope
from tests.test_helpers.builders import make_rat_spn


class _DummyOutShape:
    def __init__(self, channels: int):
        self.channels = channels


class DummyLeaf:
    def __init__(self, channels: int):
        self.out_shape = _DummyOutShape(channels=channels)
        self.scope = Scope([0])


__all__ = ["DummyLeaf", "make_rat_spn"]
