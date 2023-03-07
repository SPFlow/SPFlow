"""Contains Poisson leaf layer for SPFlow in the ``base`` backend.
"""
from typing import Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import tensorly as tl
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.base.structure.general.nodes.leaves.parametric.poisson import Poisson
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


class PoissonLayer(Module):
    r"""Layer of multiple (univariate) Poisson distribution leaf node in the ``base`` backend.

    Represents multiple univariate Poisson distributions with independent scopes, each with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \lambda^k\frac{e^{-\lambda}}{k!}

    where
        - :math:`k` is the number of occurrences
        - :math:`\lambda` is the rate parameter

    Attributes:
        l:
            One-dimensional NumPy array containing the rate parameters (:math:`\lambda`) for each of the independent Poisson distributions (greater than or equal to 0.0).
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``Poisson`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        l: Union[int, float, List[float], tl.tensor] = 1.0,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``PoissonLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            l:
                Floating point, list of floats or one-dimensional NumPy array containing the rate parameters (:math:`\lambda`) for each of the independent Poisson distributions (greater than or equal to 0.0).
                If a single floating point value is given, it is broadcast to all nodes.
                Defaults to 1.0.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'PoissonLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'PoissonLayer' was empty.")

            self._n_out = len(scope)

        super().__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [Poisson(s) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(l)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def l(self) -> tl.tensor:
        """Returns the rate parameters of the represented distributions."""
        return tl.tensor([node.l for node in self.nodes])

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``PoissonLayer`` can represent one or more univariate nodes with ``MetaType.Discrete`` or ``PoissonType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not Poisson.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "PoissonLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``PoissonLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'PoissonLayer' cannot be instantiated from the following signatures: {signatures}.")

        l = []
        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if domain == MetaType.Discrete:
                l.append(1.0)
            elif domain == FeatureTypes.Poisson:
                # instantiate object
                l.append(domain().l)
            elif isinstance(domain, FeatureTypes.Poisson):
                l.append(domain.l)
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'PoissonLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return PoissonLayer(scopes, l=l)

    def set_params(self, l: Union[int, float, List[float], tl.tensor]) -> None:
        r"""Sets the parameters for the represented distributions.

        Args:
            l:
                Floating point, list of floats or one-dimensional NumPy array containing the rate parameters (:math:`\lambda`) for each of the independent Poisson distributions (greater than or equal to 0.0).
                If a single floating point value is given, it is broadcast to all nodes.
                Defaults to 1.0.
        """
        if isinstance(l, int) or isinstance(l, float):
            l = tl.tensor([float(l) for _ in range(self.n_out)])
        if isinstance(l, list):
            l = tl.tensor(l)
        if tl.ndim(l) != 1:
            raise ValueError(
                f"Numpy array of 'l' values for 'PoissonLayer' is expected to be one-dimensional, but is {tl.ndim(l)}-dimensional."
            )
        if l.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'l' values for 'PoissonLayer' must match number of output nodes {self.n_out}, but is {l.shape[0]}"
            )

        for node_l, node in zip(l, self.nodes):
            node.set_params(node_l)

    def get_params(self) -> Tuple[tl.tensor]:
        """Returns the parameters of the represented distribution.

        Returns:
            One-dimensional NumPy array representing the rate parameters.
        """
        return (self.l,)

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

    def check_support(self, data: tl.tensor, node_ids: Optional[List[int]] = None) -> tl.tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Poisson distributions, which are:

        .. math::

            \text{supp}(\text{Poisson})=\mathbb{N}\cup\{0\}

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

        return tl.concatenate([self.nodes[i].check_support(data) for i in node_ids], axis=1)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: PoissonLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[PoissonLayer, Poisson, None]:
    r"""Structural marginalization for ``PoissonLayer`` objects in the ``base`` backend.

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
        new_node = Poisson(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = PoissonLayer(marg_scopes, *[tl.tensor(p) for p in zip(*marg_params)])
        return new_layer
