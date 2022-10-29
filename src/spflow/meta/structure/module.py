# -*- coding: utf-8 -*-
"""Contains the abstract ``MetaModule`` class for SPFlow modules.

All valid SPFlow modules should be a subclass of this class.
Custom user modules should not inherit from this class directly, but instead inherit the ``Module`` class specific to each backend.
Used for backend-agnostic typing.
"""
from abc import ABC


class MetaModule(ABC):
    """Abstract base class for Modules of any backend.

    Used for typing in backend-agnostic functions.
    """
