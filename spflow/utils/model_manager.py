"""Model serialization utilities for SPFlow modules.

This module provides functions to save and load SPFlow models using Python's
pickle protocol. These utilities enable model persistence for later use,
sharing, or deployment.
"""

import os
import pickle

from spflow.modules.module import Module

try:
    from typing import TypeAlias
except ImportError:
    TypeAlias = None


PathLike: TypeAlias = str | bytes | os.PathLike


def save_model(model: Module, path: PathLike) -> None:
    """Save an SPFlow model to disk using pickle serialization.

    Args:
        model: The SPFlow module to save.
        path: File path where the model will be saved. The file will be
            created or overwritten if it already exists.
    """
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path: PathLike) -> Module:
    """Load an SPFlow model from disk using pickle deserialization.

    Args:
        path: File path of the saved model to load.

    Returns:
        The deserialized SPFlow module.
    """
    with open(path, "rb") as file:
        return pickle.load(file)
