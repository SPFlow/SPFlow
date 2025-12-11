"""Shape representation for SPFlow modules.

Provides the ModuleShape dataclass for representing tensor shapes
(excluding batch dimension) flowing through probabilistic circuit modules.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleShape:
    """Represents tensor shape (excluding batch dimension).

    Shapes in SPFlow modules are 3-dimensional: (features, channels, repetitions).
    This dataclass provides named access and supports iteration and indexing.

    Attributes:
        features: Number of features (random variables/scope size).
        channels: Number of parallel channels/distributions.
        repetitions: Number of independent repetitions.

    Examples:
        >>> shape = ModuleShape(4, 8, 2)
        >>> shape.features
        4
        >>> shape[1]  # channels
        8
        >>> tuple(shape)
        (4, 8, 2)
    """

    features: int
    channels: int
    repetitions: int

    def __iter__(self):
        """Iterate over shape dimensions."""
        return iter((self.features, self.channels, self.repetitions))

    def __getitem__(self, idx: int) -> int:
        """Index into shape dimensions."""
        return (self.features, self.channels, self.repetitions)[idx]

    def __repr__(self) -> str:
        """Return concise string representation."""
        return f"Shape(F={self.features}, C={self.channels}, R={self.repetitions})"
