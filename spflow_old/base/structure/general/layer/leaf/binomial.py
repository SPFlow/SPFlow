"""Contains Binomial leaf layer for SPFlow in the ``base`` backend.
"""
from typing import List, Optional, Tuple, Union
from collections.abc import Iterable

import numpy as np
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.structure.spn.layer.leaf import BinomialLayer as GeneralBinomialLayer
from spflow.base.structure.general.node.leaf.binomial import Binomial
from spflow.modules.module import Module
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class BinomialLayer(Module):
    r"""Layer of multiple (univariate) Binomial distribution leaf nodes in the ``base`` backend.

    Represents multiple univariate Binomial distributions with independent scopes, each with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \binom{n}{k}p^k(1-p)^{n-k}

    where
        - :math:`p` is the success probability of each trial in :math:`[0,1]`
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Attributes:
        n:
            One-dimensional NumPy array containing the number of i.i.d. Bernoulli trials (greater or equal to 0) for each independent Binomial distribution.
        p:
            One-dimensional NumPy array containing the success probabilities for each of the independent Binomial distributions.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``Binomial`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        scope: Union[Scope, list[Scope]],
        n: Union[int, list[int], np.ndarray],
        p: Union[int, float, list[float], np.ndarray] = 0.5,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``BinomialLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            n:
                Integer, list of integers or one-dimensional NumPy array containing the number of i.i.d. Bernoulli trials (greater or equal to 0) for each independent Binomial distribution.
                If a single integer value is given it is broadcast to all nodes.
            p:
                Floating point, list of floats or one-dimensional NumPy array representing the success probabilities of the Binomial distributions between zero and one.
                If a single floating point value is given it is broadcast to all nodes.
                Defaults to 0.5.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'BinomialLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'BinomialLayer' was empty.")

            self._n_out = len(scope)

        super().__init__(children=[], **kwargs)

        self.backend = "numpy"

        # create leaf nodes
        self.nodes = [Binomial(s, 1, 0.5) for s in scope]
        for node in self.nodes:
            node.dtype = self.dtype

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(n, p)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def n(self) -> np.ndarray:
        """Returns the numbers of i.i.d. Bernoulli trials of the represented distributions."""
        return np.array([node.n for node in self.nodes])

    @property
    def p(self) -> np.ndarray:
        """Returns the success probabilities of the represented distributions."""
        return np.array([node.p for node in self.nodes], dtype=self.dtype)

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``BinomialLayer`` can represent one or more univariate nodes with ``BinomialType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not Binomial.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "BinomialLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``BinomialLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'BinomialLayer' cannot be instantiated from the following signatures: {signatures}."
            )

        n = []
        p = []
        scopes = []

        for feature_ctx in signatures:
            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if isinstance(domain, FeatureTypes.Binomial):
                n.append(domain.n)
                p.append(domain.p)
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'BinomialLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return BinomialLayer(scopes, n=n, p=p)

    def set_params(
        self,
        n: Union[int, list[int], np.ndarray],
        p: Union[int, float, list[float], np.ndarray] = 0.5,
    ) -> None:
        """Sets the parameters for the represented distributions.

        Args:
            n:
                Integer, list of integers or one-dimensional NumPy array containing the number of i.i.d. Bernoulli trials (greater or equal to 0) for each independent Binomial distribution.
                If a single integer value is given it is broadcast to all nodes.
            p:
                Floating point, list of floats or one-dimensional NumPy array representing the success probabilities of the Binomial distributions between zero and one.
                If a single floating point value is given it is broadcast to all nodes.
                Defaults to 0.5.
        """
        if isinstance(n, int):
            n = np.array([n for _ in range(self.n_out)])
        if isinstance(n, list):
            n = np.array(n)
        if n.ndim != 1:
            raise ValueError(
                f"Numpy array of 'n' values for 'BinomialLayer' is expected to be one-dimensional, but is {n.ndim}-dimensional."
            )
        if n.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'n' values for 'BinomialLayer' must match number of output nodes {self.n_out}, but is {n.shape[0]}"
            )

        if isinstance(p, int) or isinstance(p, float):
            p = np.array([float(p) for _ in range(self.n_out)], dtype=self.dtype)
        if isinstance(p, list):
            p = np.array(p, dtype=self.dtype)
        if p.ndim != 1:
            raise ValueError(
                f"Numpy array of 'p' values for 'BinomialLayer' is expected to be one-dimensional, but is {p.ndim}-dimensional."
            )
        if p.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'p' values for 'BinomialLayer' must match number of output nodes {self.n_out}, but is {p.shape[0]}"
            )

        node_scopes = np.array([s.query[0] for s in self.scopes_out])

        for node_scope in np.unique(node_scopes):
            # at least one such element exists
            n_values = n[node_scopes == node_scope]
            if not np.all(n_values == n_values[0]):
                raise ValueError(
                    "All values of 'n' for 'BinomialLayer' over the same scope must be identical."
                )

        for node_n, node_p, node in zip(n, p, self.nodes):
            node.set_parameters(node_n, node_p)

    def get_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of one-dimensional NumPy arrays representing the number of i.i.d. Bernoulli trials and success probabilities.
        """
        return self.n, self.p

    def dist(self, node_ids: Optional[list[int]] = None) -> list[rv_frozen]:
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

    def check_support(self, data: np.ndarray, node_ids: Optional[list[int]] = None) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Binomial distributions, which are:

        .. math::

            \text{supp}(\text{Binomial})=\{0,\hdots,n\}

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

    def to_dtype(self, dtype):
        self.dtype = dtype
        for node in self.nodes:
            node.dtype = self.dtype
        self.set_params(self.n, self.p.astype(dtype))


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: BinomialLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[BinomialLayer, Binomial, None]:
    """Structural marginalization for ``BinomialLayer`` objects in the ``base`` backend.

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
        new_node = Binomial(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = BinomialLayer(marg_scopes, *[np.array(p) for p in zip(*marg_params)])
        return new_layer


@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: BinomialLayer, dispatch_ctx: Optional[DispatchContext] = None):
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return GeneralBinomialLayer(scope=leaf_node.scopes_out, n=leaf_node.n, p=leaf_node.p)
