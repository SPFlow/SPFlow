from torch import Tensor

from spflow import distributions as D
from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule, log_likelihood as parent_log_likelihood
from spflow.modules.leaf.cond_leaf_module import CondLeafModule
from spflow.utils.leaf import parse_leaf_args
from typing import Callable, List, Optional, Tuple, Union
import torch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.dispatch import dispatch


class CondNormal(CondLeafModule):
    def __init__(self, scope: Scope, out_channels: int = None, cond_f: Optional[Union[Callable, list[Callable]]] = None):
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
        mean, std = self.retrieve_params(data=torch.tensor([]), dispatch_ctx=init_default_dispatch_context())
        event_shape = parse_leaf_args(scope=scope, out_channels=out_channels, params=[mean, std])
        super().__init__(scope, out_channels=event_shape[1], cond_f=cond_f)

        self.distribution = D.Normal(mean=mean, std=std, event_shape=event_shape)

    def set_cond_f(self, cond_f: Optional[Union[list[Callable], Callable]] = None) -> None:

        if isinstance(cond_f, list) and len(cond_f) != self.out_channels:
            raise ValueError(
                "'CondLeafModule' received list of 'cond_f' functions, but length does not not match number of conditional nodes."
            )

        self.cond_f = cond_f


    def retrieve_params(
            self, data: torch.Tensor, dispatch_ctx: DispatchContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Retrieves the conditional parameters of the leaf layer.

        First, checks if conditional parameters (``mean``,``std``) are passed as additional arguments in the dispatch context.
        Secondly, checks if a function or list of functions (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameter.

        Args:
            data:
                Two-dimensional NumPy array containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            Tuple of two one-dimensional NumPy array representing the means and standard deviations, respectively.

        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
        mean, std, cond_f = None, None, None

        # check dispatch cache for required conditional parameters 'mean','std'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if value for 'mean','std' specified (highest priority)
            if "mean" in args:
                mean = args["mean"]
            if "std" in args:
                std = args["std"]
            # check if alternative function to provide 'mean','std' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'mean','std' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'mean' and 'std' nor 'cond_f' is specified (via node or arguments)
        if (mean is None or std is None) and cond_f is None:
            raise ValueError(
                "'CondGaussian' requires either 'mean' and 'std' or 'cond_f' to retrieve 'mean','std' to be specified."
            )
            # ToDo: create default parameters?

        # if 'mean' or 'std' was not already specified, retrieve it
        if mean is None or std is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, list):
                mean = []
                std = []

                for f in cond_f:
                    args = f(data)
                    mean.append(args["mean"])
                    std.append(args["std"])

                mean = torch.tensor(mean).to(self.device)
                std = torch.tensor(std).to(self.device)
            else:
                args = cond_f(data)
                mean = args["mean"]
                std = args["std"]

        """
        if isinstance(mean, int) or isinstance(mean, float):
            mean = torch.tensor([mean for _ in range(self.n_out)], dtype=self.dtype, device=self.device)
        elif isinstance(mean, list) or isinstance(mean, np.ndarray):
            mean = torch.tensor(mean, dtype=self.dtype, device=self.device)
        if mean.ndim != 1:
            raise ValueError(
                f"Numpy array of 'mean' values for 'CondGaussianLayer' is expected to be one-dimensional, but is {mean.ndim}-dimensional."
            )
        if mean.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'mean' values for 'CondGaussianLayer' must match number of output nodes {self.n_out}, but is {mean.shape[0]}"
            )

        if not torch.any(torch.isfinite(mean)):
            raise ValueError(f"Values of 'mean' for 'CondGaussianLayer' must be finite, but was: {mean}")

        if isinstance(std, int) or isinstance(std, float):
            std = torch.tensor([std for _ in range(self.n_out)], dtype=self.dtype, device=self.device)
        elif isinstance(std, list) or isinstance(std, np.ndarray):
            std = torch.tensor(std, dtype=self.dtype, device=self.device)
        if std.ndim != 1:
            raise ValueError(
                f"Numpy array of 'std' values for 'CondGaussianLayer' is expected to be one-dimensional, but is {std.ndim}-dimensional."
            )
        if std.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'std' values for 'CondGaussianLayer' must match number of output nodes {self.n_out}, but is {std.shape[0]}"
            )

        if torch.any(std <= 0.0) or not torch.any(torch.isfinite(std)):
            raise ValueError(f"Value of 'std' for 'CondGaussianLayer' must be greater than 0, but was: {std}")
        """

        return mean, std

    def set_params(self, *params):
        for i, param in enumerate(params):
            if i == 0:
                self.distribution.mean.data = param
            elif i == 1:
                self.distribution.std = param
            else:
                raise ValueError(f"Too many parameters for {self.__class__.__name__}.")







