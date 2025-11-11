from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class Distribution(nn.Module, ABC):
    def __init__(self, event_shape: tuple[int, ...] = None):
        """
        Base class for all distributions.
        Args:
            event_shape: The shape of the event. If None, it is inferred from the shape of the parameter tensor.
        """
        super().__init__()

        # Check if event_shape is a tuple of positive integers
        if not all(e > 0 for e in event_shape):
            raise ValueError("Event shape must be a tuple of positive integers.")
        self.event_shape = event_shape

    @property
    @abstractmethod
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying torch distribution object."""
        pass

    @property
    @abstractmethod
    def _supported_value(self):
        """Returns the supported values of the distribution."""
        pass

    @abstractmethod
    def maximum_likelihood_estimation(
        self, data: torch.Tensor, weights: torch.Tensor = None, bias_correction: bool = True
    ):
        pass

    def sample(self, n_samples):
        """Generates samples of shape (n_samples, *event_shape)"""
        return self.distribution.sample((n_samples,))

    def mode(self):
        """Returns the mode of the distribution."""
        return self.distribution.mode

    def log_prob(self, x):
        """Computes the log probability of the given samples."""
        return self.distribution.log_prob(x)

    @property
    def out_features(self):
        """Returns the number of output features of the distribution."""
        return self.event_shape[0]

    @property
    def out_channels(self):
        """Returns the number of output channels of the distribution."""
        if len(self.event_shape) == 1:
            return 1
        else:
            return self.event_shape[1]

    @property
    def num_repetitions(self):
        """Returns the number of repetitions of the distribution."""
        if len(self.event_shape) == 3:
            return self.event_shape[2]
        else:
            return None

    @abstractmethod
    def params(self):
        """Returns the parameters of the distribution."""
        pass

    def marginalized_params(self, indices: list[int]) -> dict[str, Tensor]:
        """Returns the marginalized parameters of the distribution.

        Args:
            indices:
                List of integers specifying the indices of the module to keep.

        Returns:
            Dictionary from parameter name to tensor containing the marginalized parameters.
        """
        return {k: v[indices] for k, v in self.params().items()}

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

        # check only first entry of num_leaf node dim since all leaf node repetition have the same support

        valid[~nan_mask] = self.distribution.support.check(data)[..., [0]][~nan_mask]

        # check for infinite values
        valid[~nan_mask & valid] &= ~data[~nan_mask & valid].isinf()

        return valid
