"""Contains conditional Log-Normal leaf layer for SPFlow in the ``torch`` backend.
"""
from functools import reduce
from typing import Callable, List, Optional, Tuple, Union
from collections.abc import Iterable

import numpy as np
import torch
import tensorly as tl
import torch.distributions as D

from spflow.base.structure.general.layer.leaf.cond_log_normal import (
    CondLogNormalLayer as BaseCondLogNormalLayer,
)
from spflow.structure.general.layer.leaf import CondLogNormalLayer as GeneralCondLogNormalLayer
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.node.leaf.cond_log_normal import (
    CondLogNormal,
)
from spflow.modules.module import Module


class CondLogNormalLayer(Module):
    r"""Layer of multiple (univariate) Log-Normal distribution leaf nodes in the ``torch`` backend.

    Represents multiple univariate Log-Normal distributions with independent scopes, each with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln(x)-\mu)^2}{2\sigma^2}\right)

    where
        - :math:`x` is an observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Attributes:
        cond_f:
            Optional callable or list of callables to retrieve parameters for the leaf nodes.
            If a single callable, its output should be a dictionary contain ``mean``,``std`` as keys, and the values should be
            a floating point, a list of floats or a one-dimensional NumPy array or PyTorch tensor, containing the mean and standard deviation (the latter greater than 0), respectively.
            If the values are single floating point values, the same values are reused for all leaf nodes.
            If a list of callables, each one should return a dictionary containing ``mean``,``std`` as keys, and the values should
            be floating point values (the latter greater than 0.0).
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(
        self,
        scope: Union[Scope, list[Scope]],
        cond_f: Optional[Union[Callable, list[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``CondLogNormalLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            cond_f:
                Optional callable or list of callables to retrieve parameters for the leaf nodes.
                If a single callable, its output should be a dictionary contain ``mean``,``std`` as keys, and the values should be
                a floating point, a list of floats or a one-dimensional NumPy array or PyTorch tensor, containing the mean and standard deviation (the latter greater than 0), respectively.
                If the values are single floating point values, the same values are reused for all leaf nodes.
                If a list of callables, each one should return a dictionary containing ``mean``,``std`` as keys, and the values should
                be floating point values (the latter greater than 0.0).
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'CondLogNormalLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'CondLogNormalLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")
            if len(s.evidence) == 0:
                raise ValueError(f"Evidence scope for 'CondLogNormalLayer' should not be empty.")

        super().__init__(children=[], **kwargs)

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.join(s2), self.scopes_out)

        self.set_cond_f(cond_f)
        self.backend = "pytorch"

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CondLogNormalLayer`` can represent one or more univariate nodes with ``MetaType.Continuous`` or ``LogNormalType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not CondLogNormal.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "CondLogNormalLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``CondLogNormalLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'CondLogNormalLayer' cannot be instantiated from the following signatures: {signatures}."
            )

        scopes = []

        for feature_ctx in signatures:
            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if (
                domain == MetaType.Continuous
                or domain == FeatureTypes.LogNormal
                or isinstance(domain, FeatureTypes.LogNormal)
            ):
                pass
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'CondLogNormalLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return CondLogNormalLayer(scopes)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    def set_cond_f(self, cond_f: Optional[Union[list[Callable], Callable]] = None) -> None:
        r"""Sets the ``cond_f`` property.

        Args:
            cond_f:
                Optional callable or list of callables to retrieve parameters for the leaf nodes.
                If a single callable, its output should be a dictionary contain ``mean``,``std`` as keys, and the values should be
                a floating point, a list of floats or a one-dimensional NumPy array or PyTorch tensor, containing the mean and standard deviation (the latter greater than 0), respectively.
                If the values are single floating point values, the same values are reused for all leaf nodes.
                If a list of callables, each one should return a dictionary containing ``mean``,``std`` as keys, and the values should
                be floating point values (the latter greater than 0.0).

        Raises:
            ValueError: If list of callables does not match number of nodes represented by the layer.
        """
        if isinstance(cond_f, list) and len(cond_f) != self.n_out:
            raise ValueError(
                "'CondLogNormalLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes."
            )

        self.cond_f = cond_f

    def dist(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        node_ids: Optional[list[int]] = None,
    ) -> D.Distribution:
        r"""Returns the PyTorch distributions represented by the leaf layer.

        Args:
            mean:
                One-dimensional PyTorch array representing the means of all distributions (not just the ones specified by ``node_ids``).
            std:
                One-dimensional PyTorch array representing the standard deviations of all distributions (not just the ones specified by ``node_ids``).
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            ``torch.distributions.LogNormal`` instance.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.LogNormal(loc=mean[node_ids], scale=std[node_ids])

    def retrieve_params(
        self, data: np.ndarray, dispatch_ctx: DispatchContext
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
            Tuple of two one-dimensional PyTorch arrays representing the means and standard deviations, respectively.

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
                "'CondLogNormalLayer' requires either 'mean' and 'std' or 'cond_f' to retrieve 'mean','std' to be specified."
            )

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

                mean = torch.tensor(mean, dtype=self.dtype, device=self.device)
                std = torch.tensor(std, dtype=self.dtype, device=self.device)
            else:
                args = cond_f(data)
                mean = args["mean"]
                std = args["std"]

        if isinstance(mean, int) or isinstance(mean, float):
            mean = torch.tensor([mean for _ in range(self.n_out)], dtype=self.dtype, device=self.device)
        elif isinstance(mean, list) or isinstance(mean, np.ndarray):
            mean = torch.tensor(mean, dtype=self.dtype, device=self.device)
        if mean.ndim != 1:
            raise ValueError(
                f"Numpy array of 'mean' values for 'CondLogNormalLayer' is expected to be one-dimensional, but is {mean.ndim}-dimensional."
            )
        if mean.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'mean' values for 'CondLogNormalLayer' must match number of output nodes {self.n_out}, but is {mean.shape[0]}"
            )

        if not torch.any(torch.isfinite(mean)):
            raise ValueError(f"Values of 'mean' for 'CondLogNormalLayer' must be finite, but was: {mean}")

        if isinstance(std, int) or isinstance(std, float):
            std = torch.tensor([std for _ in range(self.n_out)], dtype=self.dtype, device=self.device)
        elif isinstance(std, list) or isinstance(std, np.ndarray):
            std = torch.tensor(std, dtype=self.dtype, device=self.device)
        if std.ndim != 1:
            raise ValueError(
                f"Numpy array of 'std' values for 'CondLogNormalLayer' is expected to be one-dimensional, but is {std.ndim}-dimensional."
            )
        if std.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'std' values for 'CondLogNormalLayer' must match number of output nodes {self.n_out}, but is {std.shape[0]}"
            )

        if torch.any(std <= 0.0) or not torch.any(torch.isfinite(std)):
            raise ValueError(
                f"Value of 'std' for 'CondLogNormalLayer' must be greater than 0, but was: {std}"
            )

        return mean, std

    def check_support(
        self,
        data: torch.Tensor,
        node_ids: Optional[list[int]] = None,
        is_scope_data: bool = False,
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Log-Normal distributions, which are:

        .. math::

            \text{supp}(\text{LogNormal})=(0,\infty)

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leafs' scope in the correct order (True) or if it needs to be extracted from the full data set.
                Note, that this should already only contain only the data according (and in order of) ``node_ids``.
                Defaults to False.

        Returns:
            Two dimensional PyTorch tensor indicating for each instance and node, whether they are part of the support (True) or not (False).
            Each row corresponds to an input sample.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        if is_scope_data:
            scope_data = data
        else:
            # all query scopes are univariate
            scope_data = data[:, [self.scopes_out[node_id].query[0] for node_id in node_ids]]

        # NaN values do not throw an error but are simply flagged as False
        valid = self.dist(torch.zeros(self.n_out), torch.ones(self.n_out), node_ids).support.check(scope_data)  # type: ignore

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # set nan_entries back to True
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf()

        return valid


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: CondLogNormalLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[CondLogNormalLayer, CondLogNormal, None]:
    """Structural marginalization for ``CondLogNormalLayer`` objects in the ``torch`` backend.

    Structurally marginalizes the specified layer module.
    If the layer's scope contains non of the random variables to marginalize, then the layer is returned unaltered.
    If the layer's scope is fully marginalized over, then None is returned.

    Args:
        layer:
            Layer module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect here. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Unaltered leaf layer or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    marginalized_node_ids = []
    marginalized_scopes = []

    for i, scope in enumerate(layer.scopes_out):
        # compute marginalized query scope
        marg_scope = [rv for rv in scope.query if rv not in marg_rvs]

        # node not marginalized over
        if len(marg_scope) == 1:
            marginalized_node_ids.append(i)
            marginalized_scopes.append(scope)

    if len(marginalized_node_ids) == 0:
        return None
    elif len(marginalized_node_ids) == 1 and prune:
        return CondLogNormal(scope=marginalized_scopes[0])
    else:
        return CondLogNormalLayer(scope=marginalized_scopes)


@dispatch(memoize=True)  # type: ignore
def toTorch(
    layer: BaseCondLogNormalLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> CondLogNormalLayer:
    """Conversion for ``CondLogNormalLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondLogNormalLayer(scope=layer.scopes_out)


@dispatch(memoize=True)  # type: ignore
def toBase(
    torch_layer: CondLogNormalLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> BaseCondLogNormalLayer:
    """Conversion for ``CondLogNormalLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondLogNormalLayer(scope=torch_layer.scopes_out)


@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: CondLogNormalLayer, dispatch_ctx: Optional[DispatchContext] = None):
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    data = tl.tensor([])
    cond_f = None
    if leaf_node.cond_f != None:
        params = leaf_node.cond_f(data)

        for key in leaf_node.cond_f(params):
            # Update the value for each key
            params[key] = tl.tensor(params[key])
        cond_f = lambda data: params
    return GeneralCondLogNormalLayer(scope=leaf_node.scopes_out, cond_f=cond_f)
