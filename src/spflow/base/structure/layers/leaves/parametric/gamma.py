# -*- coding: utf-8 -*-
"""Contains Gamma leaf layer for SPFlow in the 'base' backend.
"""
from typing import List, Union, Optional, Iterable, Tuple
import numpy as np

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.base.structure.module import Module
from spflow.base.structure.nodes.leaves.parametric.gamma import Gamma


class GammaLayer(Module):
    r"""Layer of multiple (univariate) Gamma distribution leaf nodes in the 'base' backend.

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
    def __init__(self, scope: Union[Scope, List[Scope]], alpha: Union[float, List[float], np.ndarray]=1.0, beta: Union[float, List[float], np.ndarray]=1.0, n_nodes: int=1, **kwargs) -> None:
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
                raise ValueError(f"Number of nodes for 'GammaLayer' must be greater or equal to 1, but was {n_nodes}")

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

    def set_params(self, alpha: Union[int, float, List[float], np.ndarray]=1.0, beta: Union[int, float, List[float], np.ndarray]=1.0) -> None:
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
        if(alpha.ndim != 1):
            raise ValueError(f"Numpy array of 'alpha' values for 'GammaLayer' is expected to be one-dimensional, but is {alpha.ndim}-dimensional.")
        if(alpha.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'alpha' values for 'GammaLayer' must match number of output nodes {self.n_out}, but is {alpha.shape[0]}")

        if isinstance(beta, int) or isinstance(beta, float):
            beta = np.array([float(beta) for _ in range(self.n_out)])
        if isinstance(beta, list):
            beta = np.array(beta)
        if(beta.ndim != 1):
            raise ValueError(f"Numpy array of 'beta' values for 'GammaLayer' is expected to be one-dimensional, but is {beta.ndim}-dimensional.")
        if(beta.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'beta' values for 'GammaLayer' must match number of output nodes {self.n_out}, but is {beta.shape[0]}")

        for node_mean, node_beta, node in zip(alpha, beta, self.nodes):
            node.set_params(node_mean, node_beta)
    
    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of one-dimensional NumPy arrays representing the shape and rate parameters.
        """
        return self.alpha, self.beta

    # TODO: dist

    # TODO: check support


@dispatch(memoize=True)  # type: ignore
def marginalize(layer: GammaLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[GammaLayer, Gamma, None]:
    r"""Structural marginalization for ``GammaLayer`` objects.

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
        new_layer = GammaLayer(marg_scopes, *[np.array(p) for p in zip(*marg_params)])
        return new_layer