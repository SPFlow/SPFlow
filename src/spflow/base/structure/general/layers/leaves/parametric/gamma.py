# -*- coding: utf-8 -*-
"""Contains Gamma leaf layer for SPFlow in the ``base`` backend.
"""
from typing import List, Union, Optional, Iterable, Tuple, Type
import numpy as np
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.data.scope import Scope
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.feature_types import FeatureType, FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.structure.module import Module
from spflow.base.structure.general.nodes.leaves.parametric.gamma import Gamma


class GammaLayer(Module):
    r"""Layer of multiple (univariate) Gamma distribution leaf nodes in the ``base`` backend.

    Represents multiple univariate Gamma distributions with independent scopes, each with the following probability distribution function (PDF):

    .. math::
    
        \text{PDF}(x) = \begin{cases} \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x} & \text{if } x > 0\\
                                      0 & \text{if } x <= 0\end{cases}

    where
        - :math:`x` is the input observation
        - :math:`\Gamma` is the Gamma function
        - :math:`\alpha` is the shape parameter
        - :math:`\beta` is the rate parameter
    
    Attributes:
        alpha:
            One-dimensional NumPy array representing the shape parameters (:math:`\alpha`), greater than 0.
        beta:
            One-dimensional NumPy array representing the rate parameter (:math:`\beta`), greater than 0.    
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``Gamma`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        alpha: Union[float, List[float], np.ndarray] = 1.0,
        beta: Union[float, List[float], np.ndarray] = 1.0,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``GammaLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            alpha:
                Floating point, list of floats or one-dimensional NumPy array representing the shape parameters (:math:`\alpha`), greater than 0.
                If a single floating point value is given it is broadcast to all nodes.
                Defaults to 1.0.
            beta:
                Floating point, list of floats or one-dimensional NumPy array representing the rate parameters (:math:`\beta`), greater than 0.
                If a single floating point value is given it is broadcast to all nodes.
                Defaults to 1.0.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'GammaLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'GammaLayer' was empty.")

            self._n_out = len(scope)

        super(GammaLayer, self).__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [Gamma(s) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(alpha, beta)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def alpha(self) -> np.ndarray:
        """Returns the shape parameters of the represented distributions."""
        return np.array([node.alpha for node in self.nodes])

    @property
    def beta(self) -> np.ndarray:
        """Returns the rate parameters of the represented distributions."""
        return np.array([node.beta for node in self.nodes])

    @classmethod
    def accepts(self, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``GammaLayer`` can represent one or more univariate nodes with ``MetaType.Continuous`` or ``GammaType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not Gamma.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(self, signatures: List[FeatureContext]) -> "GammaLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``GammaLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not self.accepts(signatures):
            raise ValueError(
                f"'GammaLayer' cannot be instantiated from the following signatures: {signatures}."
            )

        alpha = []
        beta = []
        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if domain == MetaType.Continuous:
                alpha.append(1.0)
                beta.append(1.0)
            elif domain == FeatureTypes.Gamma:
                # instantiate object
                alpha.append(domain().alpha)
                beta.append(domain().beta)
            elif isinstance(domain, FeatureTypes.Gamma):
                alpha.append(domain.alpha)
                beta.append(domain.beta)
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'GammaLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return GammaLayer(scopes, alpha=alpha, beta=beta)

    def set_params(
        self,
        alpha: Union[int, float, List[float], np.ndarray] = 1.0,
        beta: Union[int, float, List[float], np.ndarray] = 1.0,
    ) -> None:
        r"""Sets the parameters for the represented distributions.

        Args:
            alpha:
                Floating point, list of floats or one-dimensional NumPy array representing the shape parameters (:math:`\alpha`), greater than 0.
                If a single floating point value is given it is broadcast to all nodes.
                Defaults to 1.0.
            beta:
                Floating point, list of floats or one-dimensional NumPy array representing the rate parameters (:math:`\beta`), greater than 0.
                If a single floating point value is given it is broadcast to all nodes.
                Defaults to 1.0.
        """
        if isinstance(alpha, int) or isinstance(alpha, float):
            alpha = np.array([alpha for _ in range(self.n_out)])
        if isinstance(alpha, list):
            alpha = np.array(alpha)
        if alpha.ndim != 1:
            raise ValueError(
                f"Numpy array of 'alpha' values for 'GammaLayer' is expected to be one-dimensional, but is {alpha.ndim}-dimensional."
            )
        if alpha.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'alpha' values for 'GammaLayer' must match number of output nodes {self.n_out}, but is {alpha.shape[0]}"
            )

        if isinstance(beta, int) or isinstance(beta, float):
            beta = np.array([float(beta) for _ in range(self.n_out)])
        if isinstance(beta, list):
            beta = np.array(beta)
        if beta.ndim != 1:
            raise ValueError(
                f"Numpy array of 'beta' values for 'GammaLayer' is expected to be one-dimensional, but is {beta.ndim}-dimensional."
            )
        if beta.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'beta' values for 'GammaLayer' must match number of output nodes {self.n_out}, but is {beta.shape[0]}"
            )

        for node_mean, node_beta, node in zip(alpha, beta, self.nodes):
            node.set_params(node_mean, node_beta)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of one-dimensional NumPy arrays representing the shape and rate parameters.
        """
        return self.alpha, self.beta

    def dist(self, node_ids: Optional[List[int]] = None) -> List[rv_frozen]:
        r"""Returns the SciPy distributions represented by the leaf layer.

        Args:
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            List of ``scipy.stats.distributions.rv_frozen`` distributions.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return [self.nodes[i].dist for i in node_ids]

    def check_support(
        self, data: np.ndarray, node_ids: Optional[List[int]] = None
    ) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Gamma distributions, which are:

        .. math::

            \text{supp}(\text{Gamma})=(0,+\infty)

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

        return np.concatenate(
            [self.nodes[i].check_support(data) for i in node_ids], axis=1
        )


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: GammaLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[GammaLayer, Gamma, None]:
    r"""Structural marginalization for ``GammaLayer`` objects in the ``base`` backend.

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
    marg_params = []

    for node in layer.nodes:
        marg_node = marginalize(node, marg_rvs, prune=prune)

        if marg_node is not None:
            marg_scopes.append(marg_node.scope)
            marg_params.append(marg_node.get_params())

    if len(marg_scopes) == 0:
        return None
    elif len(marg_scopes) == 1 and prune:
        new_node = Gamma(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = GammaLayer(
            marg_scopes, *[np.array(p) for p in zip(*marg_params)]
        )
        return new_layer
