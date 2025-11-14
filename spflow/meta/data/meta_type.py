"""Meta types for data feature types."""

from enum import Enum


class MetaType(Enum):
    """Enum containing meta types underlying any data feature type.

    Attributes:
        Unknown: Unknown feature type.
        Continuous: Continuous feature type.
        Discrete: Discrete feature type.
    """

    Unknown: int = -1
    Continuous: int = 0
    Discrete: int = 1

    @classmethod
    def get_params(cls) -> tuple:
        """Returns an empty tuple of parameters.

        Returns:
            tuple: Empty tuple.
        """
        return tuple([])
