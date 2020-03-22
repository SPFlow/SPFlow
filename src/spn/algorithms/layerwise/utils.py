#!/usr/bin/env python3
from contextlib import contextmanager

import torch
from torch import nn

from spn.algorithms.layerwise import layers


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
            if isinstance(module, layers.Sum):
                module._enable_sampling_input_cache()

        if evidence is not None:
            _ = spn(evidence)

        # Run in context (nothing needs to be yielded)
        yield

        # Exit
        for module in spn.modules():
            if isinstance(module, layers.Sum):
                module._disable_sampling_input_cache()
