#!/usr/bin/env python3

from typing import Iterable
from torch import Tensor, nn
import torch
from abc import ABC, abstractmethod


class Distribution(nn.Module, ABC):
    def __init__(self, event_shape: tuple[int, ...] = None):
        super().__init__()

        # Check if event_shape is a tuple of positive integers
        assert all(e > 0 for e in event_shape), "Event shape must be a tuple of positive integers."
        self.event_shape = event_shape

    @property
    @abstractmethod
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying torch distribution object."""
        pass

    @abstractmethod
    def maximum_likelihood_estimation(
        self, data: torch.Tensor, weights: torch.Tensor = None, bias_correction: bool = True
    ):
        pass

    def sample(self, n_samples):
        """Generates samples of shape (n_samples, *event_shape)"""
        return self.distribution.sample((n_samples,))

    @abstractmethod
    def mode(self):
        """Returns the mode of the distribution."""
        pass

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    @abstractmethod
    def marginalized_params(self, indices: list[int]) -> dict[str, Tensor]:
        """Returns the marginalized parameters of the distribution.

        Args:
            indices:
                List of integers specifying the indices of the module to keep.

        Returns:
            Dictionary from parameter name to tensor containing the marginalized parameters.
        """
        pass

    @abstractmethod
    def accepts(self, signatures):
        """Checks if the leaf node accepts the given signatures.

        Args:
            signatures:
                List of FeatureContext objects representing the signatures to check.

        Returns:
            Boolean indicating if the leaf node accepts the given signatures.
        """
        pass

    @abstractmethod
    def from_signatures(self, signatures):
        """Creates a new leaf node from the given signatures.

        Args:
            signatures:
                List of FeatureContext objects representing the signatures to create the leaf node from.

        Returns:
            A new leaf node created from the given signatures.
        """
        pass

    def check_support(self, data: Tensor) -> Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of this distribution.

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.

        Returns:
            Two dimensional PyTorch tensor indicating for each instance, whether they are part of the support (True) or not (False).
        """

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(data)

        valid = torch.ones_like(data, dtype=torch.bool)
        valid[~nan_mask] = self.distribution.support.check(data[~nan_mask])

        # check for infinite values
        valid[~nan_mask & valid] &= ~data[~nan_mask & valid].isinf()

        return valid
