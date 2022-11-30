"""Contains conditional Log-Normal leaf layer for SPFlow in the ``base`` backend.
"""
from typing import Callable, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.base.structure.general.nodes.leaves.parametric.cond_log_normal import (
    CondLogNormal,
)
from spflow.base.structure.module import Module
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureType, FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class CondLogNormalLayer(Module):
    r"""Layer of multiple (univariate) Log-Normal distribution leaf nodes in the ``base`` backend.

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
            a floating point, a list of floats or a one-dimensional NumPy array, containing the mean and standard deviation (the latter greater than 0), respectively.
            If the values are single floating point values, the same values are reused for all leaf nodes.
            If a list of callables, each one should return a dictionary containing ``mean``,``std`` as keys, and the values should
            be floating point values (the latter greater than 0.0).
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``CondLogNormal`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        cond_f: Optional[Union[Callable, List[Callable]]] = None,
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
                a floating point, a list of floats or a one-dimensional NumPy array, containing the mean and standard deviation (the latter greater than 0), respectively.
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

        super().__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [CondLogNormal(s) for s in scope]

        # compute scope
        self.scopes_out = scope

        self.set_cond_f(cond_f)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
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
    def from_signatures(cls, signatures: List[FeatureContext]) -> "CondLogNormalLayer":
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

    def set_cond_f(self, cond_f: Optional[Union[List[Callable], Callable]] = None) -> None:
        r"""Sets the ``cond_f`` property.

        Args:
            cond_f:
                Optional callable or list of callables to retrieve parameters for the leaf nodes.
                If a single callable, its output should be a dictionary contain ``mean``,``std`` as keys, and the values should be
                a floating point, a list of floats or a one-dimensional NumPy array, containing the mean and standard deviation (the latter greater than 0), respectively.
                If the values are single floating point values, the same values are reused for all leaf nodes.
                If a list of callables, each one should return a dictionary containing ``mean``,``std`` as keys, and the values should
                be floating point values (the latter greater than 0.0).

        Raises:
            ValueError: If list of callables does not match number of nodes represented by the layer.
        """
        if isinstance(cond_f, List) and len(cond_f) != self.n_out:
            raise ValueError(
                "'CondLogNormalLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes."
            )

        self.cond_f = cond_f

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> Tuple[np.ndarray, np.ndarray]:
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
            Tuple of two one-dimensional NumPy arrays representing the means and standard deviations, respectively.

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
            if isinstance(cond_f, List):
                mean = []
                std = []

                for f in cond_f:
                    args = f(data)
                    mean.append(args["mean"])
                    std.append(args["std"])

                mean = np.array(mean)
                std = np.array(std)
            else:
                args = cond_f(data)
                mean = args["mean"]
                std = args["std"]

        if isinstance(mean, int) or isinstance(mean, float):
            mean = np.array([mean for _ in range(self.n_out)])
        if isinstance(mean, list):
            mean = np.array(mean)
        if mean.ndim != 1:
            raise ValueError(
                f"Numpy array of 'mean' values for 'CondLogNormalLayer' is expected to be one-dimensional, but is {mean.ndim}-dimensional."
            )
        if mean.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'mean' values for 'CondLogNormalLayer' must match number of output nodes {self.n_out}, but is {mean.shape[0]}"
            )

        if isinstance(std, int) or isinstance(std, float):
            std = np.array([float(std) for _ in range(self.n_out)])
        if isinstance(std, list):
            std = np.array(std)
        if std.ndim != 1:
            raise ValueError(
                f"Numpy array of 'std' values for 'CondLogNormalLayer' is expected to be one-dimensional, but is {std.ndim}-dimensional."
            )
        if std.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'std' values for 'CondLogNormalLayer' must match number of output nodes {self.n_out}, but is {std.shape[0]}"
            )

        return mean, std

    def dist(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        node_ids: Optional[List[int]] = None,
    ) -> List[rv_frozen]:
        r"""Returns the SciPy distributions represented by the leaf layer.

        Args:
            mean:
                One-dimensional NumPy array representing the means of all distributions (not just the ones specified by ``node_ids``).
            std:
                One-dimensional NumPy array representing the standard deviations of all distributions greater than 0.0 (not just the ones specified by ``node_ids``).
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            List of ``scipy.stats.distributions.rv_frozen`` distributions.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return [self.nodes[i].dist(mean[i], std[i]) for i in node_ids]

    def check_support(self, data: np.ndarray, node_ids: Optional[List[int]] = None) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Log-Normal distributions, which are:

        .. math::

            \text{supp}(\text{LogNormal})=(0,\infty)

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional NumPy array containing sample instances.
                Each row is regarded as a sample.
                Assumes that relevant data is located in the columns corresponding to the scope indices.
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            Two dimensional NumPy array indicating for each instance and node, whether they are part of the support (True) or not (False).
            Each row corresponds to an input sample.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return np.concatenate([self.nodes[i].check_support(data) for i in node_ids], axis=1)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: CondLogNormalLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[CondLogNormalLayer, CondLogNormal, None]:
    r"""Structural marginalization for ``CondLogNormalLayer`` objects in the ``base`` backend.

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

    # marginalize nodes
    marg_scopes = []

    for node in layer.nodes:
        marg_node = marginalize(node, marg_rvs, prune=prune)

        if marg_node is not None:
            marg_scopes.append(marg_node.scope)

    if len(marg_scopes) == 0:
        return None
    elif len(marg_scopes) == 1 and prune:
        new_node = CondLogNormal(marg_scopes[0])
        return new_node
    else:
        new_layer = CondLogNormalLayer(marg_scopes)
        return new_layer
