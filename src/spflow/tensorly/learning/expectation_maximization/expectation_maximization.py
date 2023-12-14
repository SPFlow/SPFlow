"""Contains the expectation maximization optimization parameter learner for SPFlow in the ``torch`` backend.
"""
from typing import List

import torch
import tensorly as tl
from spflow.tensorly.utils.helper_functions import T

from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensorly.inference.module import log_likelihood
#from spflow.torch.learning.general.node.leaf.bernoulli import em  # TODO
from spflow.torch.learning import em
from spflow.tensorly.structure.module import Module


def expectation_maximization(
    module: Module,
    data: T,
    max_steps: int = -1,
    check_support: bool = True,
) -> T:
    """Performs partitioning usig randomized dependence coefficients (RDCs) to be used with the LearnSPN algorithm in the ``torch`` backend.

    Args:
        module:
            Module to perform EM optimization on.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        max_steps:
            Integer representing the maximum number of iterations.
            Defaults to -1, in which case the optimization is performed until convergence.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.

    Returns:
        One-dimensional PyTorch tensors, containing the average log-likelihood for each iteration step.
    """
    prev_avg_ll = tl.tensor(-float("inf"))
    ll_history = []
    backend = tl.get_backend()

    while True:

        # initialize new dispatch context
        dispatch_ctx = DispatchContext()

        # compute log likelihoods and sum them together
        acc_ll = log_likelihood(module, data, check_support=check_support, dispatch_ctx=dispatch_ctx).sum()
        if backend=="pytorch":
            avg_ll = acc_ll.detach().clone() / data.shape[0]
        else:
            avg_ll = acc_ll / data.shape[0]

        ll_history.append(avg_ll)

        # end update loop if max steps reached or loss converged
        if max_steps == 0 or avg_ll <= prev_avg_ll:
            break

        # retain gradients for all module log-likelihoods
        if backend=="pytorch":
            for ll in dispatch_ctx.cache["log_likelihood"].values():
                if ll.requires_grad:
                    ll.retain_grad()

            # compute gradients (if there are differentiable parameters to begin with)
            if acc_ll.requires_grad:
                acc_ll.backward()

        # recursively perform expectation maximization
        em(module, data, check_support=check_support, dispatch_ctx=dispatch_ctx)

        prev_avg_ll = avg_ll
        # TODO: zero/None all gradients

        # increment steps counter
        if max_steps > 0:
            max_steps -= 1

    return tl.stack(ll_history)
