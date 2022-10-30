# -*- coding: utf-8 -*-
"""Contains Hypergeometric leaf layer for SPFlow in the ``base`` backend.
"""
from typing import List, Union, Optional, Iterable, Tuple
import numpy as np
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.data.scope import Scope
from spflow.base.structure.module import Module
from spflow.base.structure.nodes.leaves.parametric.hypergeometric import (
    Hypergeometric,
)


class HypergeometricLayer(Module):
    r"""Layer of multiple (univariate) Hypergeometric distribution leaf node in the ``base`` backend.

    Represents multiple univariate Hypergeometric distributions with independent scopes, each with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \frac{\binom{M}{k}\binom{N-M}{n-k}}{\binom{N}{n}}

    where
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)
        - :math:`N` is the total number of entities
        - :math:`M` is the number of entities with property of interest
        - :math:`n` is the number of draws
        - :math:`k` s the number of observed entities

    Attributes:
        N:
            One-dimensional NumPy array specifying the total numbers of entities (in the populations), greater or equal to 0.
        M:
            One-dimensional NumPy array specifying the numbers of entities with property of interest (in the populations), greater or equal to zero and less than or equal to N.
        n:
            One-dimensional NumPy array specifying the numbers of draws, greater of equal to zero and less than or equal to N.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``Hypergeometric`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        N: Union[int, List[int], np.ndarray],
        M: Union[int, List[int], np.ndarray],
        n: Union[int, List[int], np.ndarray],
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``HypergeometricLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            N:
                Integer, list of ints or one-dimensional NumPy array specifying the total numbers of entities (in the populations), greater or equal to 0.
                If a single integer value is given it is broadcast to all nodes.
            M:
                Integer, list of ints or one-dimensional NumPy array specifying the numbers of entities with property of interest (in the populations), greater or equal to zero and less than or equal to N.
                If a single integer value is given it is broadcast to all nodes.
            n:
                Integer, list of ints or one-dimensional NumPy array specifying the numbers of draws, greater of equal to zero and less than or equal to N.
                If a single integer value is given it is broadcast to all nodes.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'HypergeometricLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError(
                    "List of scopes for 'HypergeometricLayer' was empty."
                )

            self._n_out = len(scope)

        super(HypergeometricLayer, self).__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [Hypergeometric(s, 1, 1, 1) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(N, M, n)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def N(self) -> np.ndarray:
        """Returns the total numbers of entities (in the populations)"""
        return np.array([node.N for node in self.nodes])

    @property
    def M(self) -> np.ndarray:
        """Returns the numbers of entities with property of interest (in the populations)."""
        return np.array([node.M for node in self.nodes])

    @property
    def n(self) -> np.ndarray:
        """Returns the numbers of draws."""
        return np.array([node.n for node in self.nodes])

    def set_params(
        self,
        N: Union[int, List[int], np.ndarray],
        M: Union[int, List[int], np.ndarray],
        n: Union[int, List[int], np.ndarray],
    ) -> None:
        """Sets the parameters for the represented distributions.

        Args:
            N:
                Integer, list of ints or one-dimensional NumPy array specifying the total numbers of entities (in the populations), greater or equal to 0.
                If a single integer value is given it is broadcast to all nodes.
            M:
                Integer, list of ints or one-dimensional NumPy array specifying the numbers of entities with property of interest (in the populations), greater or equal to zero and less than or equal to N.
                If a single integer value is given it is broadcast to all nodes.
            n:
                Integer, list of ints or one-dimensional NumPy array specifying the numbers of draws, greater of equal to zero and less than or equal to N.
                If a single integer value is given it is broadcast to all nodes.
        """
        if isinstance(N, int):
            N = np.array([N for _ in range(self.n_out)])
        if isinstance(N, list):
            N = np.array(N)
        if N.ndim != 1:
            raise ValueError(
                f"Numpy array of 'N' values for 'HypergeometricLayer' is expected to be one-dimensional, but is {N.ndim}-dimensional."
            )
        if N.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'N' values for 'HypergeometricLayer' must match number of output nodes {self.n_out}, but is {N.shape[0]}"
            )

        if isinstance(M, int):
            M = np.array([M for _ in range(self.n_out)])
        if isinstance(M, list):
            M = np.array(M)
        if M.ndim != 1:
            raise ValueError(
                f"Numpy array of 'M' values for 'HypergeometricLayer' is expected to be one-dimensional, but is {M.ndim}-dimensional."
            )
        if M.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'M' values for 'HypergeometricLayer' must match number of output nodes {self.n_out}, but is {M.shape[0]}"
            )

        if isinstance(n, int):
            n = np.array([n for _ in range(self.n_out)])
        if isinstance(n, list):
            n = np.array(n)
        if n.ndim != 1:
            raise ValueError(
                f"Numpy array of 'n' values for 'HypergeometricLayer' is expected to be one-dimensional, but is {n.ndim}-dimensional."
            )
        if n.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'n' values for 'HypergeometricLayer' must match number of output nodes {self.n_out}, but is {n.shape[0]}"
            )

        node_scopes = np.array([s.query[0] for s in self.scopes_out])

        for node_scope in np.unique(node_scopes):
            # at least one such element exists
            N_values = N[node_scopes == node_scope]
            if not np.all(N_values == N_values[0]):
                raise ValueError(
                    "All values of 'N' for 'HypergeometricLayer' over the same scope must be identical."
                )
            # at least one such element exists
            M_values = M[node_scopes == node_scope]
            if not np.all(M_values == M_values[0]):
                raise ValueError(
                    "All values of 'M' for 'HypergeometricLayer' over the same scope must be identical."
                )
            # at least one such element exists
            n_values = n[node_scopes == node_scope]
            if not np.all(n_values == n_values[0]):
                raise ValueError(
                    "All values of 'n' for 'HypergeometricLayer' over the same scope must be identical."
                )

        for node_N, node_M, node_n, node in zip(N, M, n, self.nodes):
            node.set_params(node_N, node_M, node_n)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the parameters of the represented distribution.

        Returns:
            Thee one-dimensional NumPy arrays representing the total numbers of entities, the numbers of entities of interest and the numbers of draws.
        """
        return self.N, self.M, self.n

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

        Determines whether or note instances are part of the supports of the Hypergeometric distributions, which are:

        .. math::

            \text{supp}(\text{Hypergeometric})={\max(0,n+M-N),...,\min(n,M)}

        where
            - :math:`N` is the total number of entities
            - :math:`M` is the number of entities with property of interest
            - :math:`n` is the number of draws

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
    layer: HypergeometricLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[HypergeometricLayer, Hypergeometric, None]:
    """Structural marginalization for ``HypergeometricLayer`` objects in the ``base`` backend.

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
        new_node = Hypergeometric(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = HypergeometricLayer(
            marg_scopes, *[np.array(p) for p in zip(*marg_params)]
        )
        return new_layer
