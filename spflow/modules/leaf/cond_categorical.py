from __future__ import annotations

from torch import Tensor

from spflow import distributions as D
from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule, log_likelihood as parent_log_likelihood
from spflow.modules.leaf.cond_leaf_module import CondLeafModule
from spflow.utils.leaf import parse_leaf_args
from collections.abc import Callable
import torch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.dispatch import dispatch


class CondCategorical(CondLeafModule):
    def __init__(self, scope: Scope, out_channels: int = None, K: Tensor = None, cond_f: Callable | list[Callable] | None = None):
        """
        Initialize a Normal distribution leaf module.

        Args:
            scope (Scope): The scope of the distribution.
            out_channels (int, optional): The number of output channels. If None, it is determined by the parameter tensors.
            mean (Tensor, optional): The mean parameter tensor.
            std (Tensor, optional): The standard deviation parameter tensor.
        """
        if out_channels is None and cond_f is None:
            raise ValueError("out_channels or cond_f must be provided.")
        self.set_cond_f(cond_f)
        p = self.retrieve_params(data=torch.tensor([]), dispatch_ctx=init_default_dispatch_context())
        event_shape = parse_leaf_args(scope=scope, out_channels=out_channels, params=[p])
        super().__init__(scope, out_channels=event_shape[1])
        self.distribution = D.Categorical(p, K=K, event_shape=event_shape)

    def set_cond_f(self, cond_f: list[Callable] | Callable | None = None) -> None:

        if isinstance(cond_f, list) and len(cond_f) != self.out_channels:
            raise ValueError(
                "'CondLeafModule' received list of 'cond_f' functions, but length does not not match number of conditional nodes."
            )

        self.cond_f = cond_f

    def retrieve_params(self, data: Tensor, dispatch_ctx: DispatchContext) -> torch.Tensor:
        r"""Retrieves the conditional parameters of the leaf layer.

        First, checks if conditional parameter (``p``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function or list of functions (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameter.

        Args:
            data:
                Two-dimensional PyTorch tensor containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            One-dimensional PyTorch tensor representing the success probabilities.

        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
        p, cond_f = None, None

        # check dispatch cache for required conditional parameter 'p'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if a value for 'p' is specified (highest priority)
            if "p" in args:
                p = args["p"]
            # check if alternative function to provide 'p' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'p' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'p' nor 'cond_f' is specified (via node or arguments)
        if p is None and cond_f is None:
            raise ValueError(
                "'CondBernoulliLayer' requires either 'p' or 'cond_f' to retrieve 'p' to be specified."
            )

        # if 'p' was not already specified, retrieve it
        if p is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, list):
                p = torch.tensor([f(data)["p"] for f in cond_f], dtype=self.dtype, device=self.device)
            else:
                p = cond_f(data)["p"]

        return p

    def set_params(self, *params):
        for i, param in enumerate(params):
            if i == 0:
                self.distribution.p = param
            else:
                raise ValueError(f"Too many parameters for {self.__class__.__name__}.")







