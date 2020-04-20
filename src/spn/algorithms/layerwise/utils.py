#!/usr/bin/env python3
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


@contextmanager
def provide_evidence(spn: nn.Module, evidence: torch.Tensor):
    """
    Context manager for sampling with evidence. In this context, the SPN graph is reweighted with the likelihoods
    computed using the given evidence.

    Args:
        spn: SPN that is being used to perform the sampling.
        evidence: Provided evidence. The SPN will perform a forward pass prior to entering this contex.
    """
    with torch.no_grad():
        # Enter
        for module in spn.modules():
            if hasattr(module, "_enable_sampling_input_cache"):
                module._enable_sampling_input_cache()

        if evidence is not None:
            _ = spn(evidence)

        # Run in context (nothing needs to be yielded)
        yield

        # Exit
        for module in spn.modules():
            if hasattr(module, "_enable_sampling_input_cache"):
                module._disable_sampling_input_cache()


@dataclass
class SamplingContext:
    # Number of samples
    n: int = None

    # Indices into the out_channels dimension
    parent_indices: torch.Tensor = None

    # Indices into the repetition dimension
    repetition_indices: torch.Tensor = None

    def __setattr__(self, key, value):
        if hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"SamplingContext object has no attribute {key}")
