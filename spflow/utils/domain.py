"""Domain and DataType utilities for probabilistic circuits.

This module provides classes for describing feature domains, including
whether features are continuous or discrete and their valid ranges.
"""

from enum import Enum
from typing import List, Optional

import numpy as np


class DataType(str, Enum):
    """Enum for the type of the data."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class Domain:
    """Describes the domain of a random variable.

    Attributes:
        values: List of valid discrete values (for discrete domains).
        min: Minimum value in the domain.
        max: Maximum value in the domain.
        data_type: Whether the domain is continuous or discrete.
    """

    def __init__(
        self,
        values: Optional[List] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        data_type: Optional[DataType] = None,
    ):
        """Initialize a Domain.

        Args:
            values: List of valid discrete values.
            min: Minimum value in the domain.
            max: Maximum value in the domain.
            data_type: DataType enum specifying continuous or discrete.
        """
        self.values = values
        self.min = min
        self.max = max
        self.data_type = data_type

    @staticmethod
    def discrete_bins(values: List) -> "Domain":
        """Create a discrete domain from a list of valid values.

        Args:
            values: List of valid discrete values.

        Returns:
            Domain with discrete data type.
        """
        return Domain(
            min=min(values),
            max=max(values),
            values=values,
            data_type=DataType.DISCRETE,
        )

    @staticmethod
    def discrete_range(min_val: int, max_val: int) -> "Domain":
        """Create a discrete domain from a range of integers.

        Args:
            min_val: Minimum value (inclusive).
            max_val: Maximum value (inclusive).

        Returns:
            Domain with discrete data type.
        """
        return Domain(
            min=min_val,
            max=max_val,
            values=list(range(min_val, max_val + 1)),
            data_type=DataType.DISCRETE,
        )

    @staticmethod
    def continuous_range(min_val: float, max_val: float) -> "Domain":
        """Create a continuous domain from a range.

        Args:
            min_val: Minimum value.
            max_val: Maximum value.

        Returns:
            Domain with continuous data type.
        """
        return Domain(min=min_val, max=max_val, data_type=DataType.CONTINUOUS)

    @staticmethod
    def continuous_inf_support() -> "Domain":
        """Create a continuous domain with infinite support.

        Returns:
            Domain with continuous data type and infinite bounds.
        """
        return Domain(min=-np.inf, max=np.inf, data_type=DataType.CONTINUOUS)

    def __repr__(self) -> str:
        if self.data_type == DataType.DISCRETE:
            return f"Domain(discrete, values={self.values})"
        else:
            return f"Domain(continuous, min={self.min}, max={self.max})"
