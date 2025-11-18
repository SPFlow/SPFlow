from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor


class Distribution(nn.Module, ABC):
    def __init__(self, event_shape: tuple[int, ...] = None):
        """
        Base class for all distributions.

        A distribution implementation has the following tasks:
        - Define the underlying torch.distributions.Distribution object.
        - Implement the maximum likelihood estimation (MLE) update statistics method (_mle_update_statistics).
        - Provide access to the distribution parameters via the params() method.
        - Define the supported values of the distribution via the _supported_value property.

        Args:
            event_shape: The shape of the event. If None, it is inferred from the shape of the parameter tensor.
        """
        super().__init__()

        # Check if event_shape is a tuple of positive integers
        assert all(e > 0 for e in event_shape), "Event shape must be a tuple of positive integers."
        self.event_shape = event_shape

    @property
    @abstractmethod
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying torch distribution object."""
        pass

    @property
    @abstractmethod
    def _supported_value(self) -> float:
        """Returns the supported values of the distribution."""
        pass

    @abstractmethod
    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool):
        """Compute distribution-specific statistics and assign parameters.

        Hook method called by maximum_likelihood_estimation() after data preparation.
        Descriptors handle validation and storage automatically.

        Args:
            data: Scope-filtered data.
            weights: Normalized weights.
            bias_correction: Apply bias correction.
        """
        pass

    @property
    def device(self) -> torch.device:
        """Return device of first parameter or buffer.

        Returns:
            Device of the module.
        """
        try:
            return next(iter(self.parameters())).device
        except StopIteration:
            return next(iter(self.buffers())).device

    def sample(self, n_samples):
        """Generates samples of shape (n_samples, *event_shape)"""
        if isinstance(n_samples, tuple):
            return self.distribution.sample(n_samples)
        else:
            return self.distribution.sample((n_samples,))

    @property
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
    def params(self) -> dict[str, Tensor]:
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

        # check only the first entry of num_leaf node dim since all leaf node repetitions have the same support

        valid[~nan_mask] = self.distribution.support.check(data)[..., [0]][~nan_mask]

        # check for infinite values
        valid[~nan_mask & valid] &= ~data[~nan_mask & valid].isinf()

        return valid

    def _broadcast_to_event_shape(self, param_est: Tensor) -> Tensor:
        """Broadcast parameter estimate to match event_shape.

        Args:
            param_est: Parameter estimate tensor to broadcast.

        Returns:
            Parameter estimate broadcasted to match event_shape.
        """
        if len(self.event_shape) == 2:
            param_est = param_est.unsqueeze(1).repeat(
                1,
                self.out_channels,
                *([1] * (param_est.dim() - 1)),
            )
        elif len(self.event_shape) == 3:
            param_est = (
                param_est.unsqueeze(1)
                .unsqueeze(1)
                .repeat(
                    1,
                    self.out_channels,
                    self.num_repetitions,
                    *([1] * (param_est.dim() - 1)),
                )
            )

        return param_est
