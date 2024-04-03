#!/usr/bin/env python3

import torch
from torch import Tensor
from spflow.distributions.distribution import Distribution
from spflow.meta.data import FeatureContext, FeatureTypes


class Uniform(Distribution):
    def __init__(self, start: Tensor, end: Tensor, support_outside: Tensor = torch.tensor(True),
                 event_shape: tuple[int, ...] = None):
        r"""Initializes ``Uniform`` leaf node.

        Args:
            scope: Scope object specifying the scope of the distribution.
            start: PyTorch tensor containing the start of the intervals (including).
            end: PyTorch tensor containing the end of the intervals (including). Must be larger than 'start'.
            end_next:
                PyTorch tensor containing the next largest floating point values to ``end``.
                Used for the PyTorch distributions which do not include the specified ends of the intervals.
            support_outside:
                PyTorch tensor containing booleans indicating whether or not values outside of the intervals are part of the support.
            n_out: Number of nodes per scope. Only relevant if mean and std is None.
        """
        if event_shape is None:
            event_shape = start.shape
        super().__init__(event_shape=event_shape)

        # register interval bounds as torch buffers (should not be changed)
        self.register_buffer("start", torch.empty(size=[]))
        self.register_buffer("end", torch.empty(size=[]))
        self.register_buffer("end_next", torch.empty(size=[]))
        self.register_buffer("support_outside", torch.empty(size=[]))

        self.check_inputs(start, end, support_outside)

        self.start = start
        self.end = end
        self.end_next = torch.nextafter(end, torch.tensor(float("inf")))
        self.support_outside = torch.tensor(support_outside)

    def check_inputs(self, start: Tensor, end: Tensor, support_outside: torch.Tensor):
        if not torch.any(torch.isfinite(start)):
            raise ValueError(f"Values of 'start' for a uniform distribution must be finite, but was: {start}")
        if not torch.any(torch.isfinite(end)):
            raise ValueError(f"Values of 'end' for a uniform distribution must be finite, but was: {end}")
        if not (start < end).all():
            raise ValueError(f"Start must be smaller than end. Got start={start} and end={end}.")


    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Uniform(self.start, self.end)

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        feature_ctx = signatures[0]
        domains = feature_ctx.get_domains()

        # leaf is a single non-conditional univariate node
        if (
                len(domains) != 1
                or len(feature_ctx.scope.query) != len(domains)
                or len(feature_ctx.scope.evidence) != 0
        ):
            return False

        # leaf is a continuous Uniform distribution
        # NOTE: only accept instances of 'FeatureTypes.Uniform', otherwise required parameters 'start','end' are not specified. Reject 'FeatureTypes.Continuous' for the same reason.
        if not isinstance(domains[0], FeatureTypes.Uniform):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Uniform":
        if not cls.accepts(signatures):
            raise ValueError(f"'Uniform' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if isinstance(domain, FeatureTypes.Uniform):
            start, end = domain.start, domain.end
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Uniform' that was not caught during acception checking."
            )

        return Uniform(start=start, end=end)

    def maximum_likelihood_estimation(self, data: Tensor, weights: Tensor = None, bias_correction=True):
        """
        All parameters of the Uniform distribution are regarded as fixed and will not be estimated.
        Therefore, this method does nothing, but check for the validity of the data.
        """
        data = data.unsqueeze(2)
        if torch.any(~ self.check_support(data)):
            raise ValueError("Encountered values outside of the support for uniform distribution.")

        # do nothing since there are no learnable parameters
        pass

    def check_support(
            self,
            data: torch.Tensor,
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.
        """

        # torch distribution support is an interval, despite representing a distribution over a half-open interval
        # end is adjusted to the next largest number to make sure that desired end is part of the distribution interval
        # may cause issues with the support check; easier to do a manual check instead
        valid = torch.ones(data.shape, dtype=torch.bool)

        # check if values are within valid range
        # check only first entry of num_leaf node dim since all leaf node repetition have the same support
        valid &= ((data >= self.start) & (
                data < self.end
        ))[..., [0]]
        valid |= self.support_outside

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(data)
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~(data[~nan_mask & valid].isinf())

        return valid

    def marginalized_params(self, indices: list[int]) -> dict[str, Tensor]:
        return {"start": self.start[indices], "end": self.end[indices]}
