"""Meta types for data feature types.
"""
from enum import Enum
from typing import Tuple


class MetaType(Enum):
    """Enum containing meta types underlying any data feature type."""

    Unknown: int = -1
    Continuous: int = 0
    Discrete: int = 1

    @classmethod
    def get_params(cls) -> tuple:
        return tuple([])
