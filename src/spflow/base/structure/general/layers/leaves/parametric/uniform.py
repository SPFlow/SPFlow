"""Contains conditional Uniform leaf node for SPFlow in the ``base`` backend.
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
from spflow.base.structure.general.nodes.leaves.parametric.uniform import (
    Uniform,
)


class UniformLayer(Module):
    r"""Layer of multiple (univariate) continuous Uniform distribution leaf nodes in the ``base`` backend.

    Represents multiple univariate Poisson distributions with independent scopes, each with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\text{end} - \text{start}}\mathbf{1}_{[\text{start}, \text{end}]}(x)

    where
        - :math:`x` is the input observation
        - :math:`\mathbf{1}_{[\text{start}, \text{end}]}` is the indicator function for the given interval (evaluating to 0 if x is not in the interval)

    Attributes:
        start:
            One-dimensional NumPy array containing the start of the intervals (including).
        end:
            One-dimensional NumPy array containing the end of the intervals (including). Must be larger than 'start'.
        support_outside:
            One-dimensional NumPy array containing booleans indicating whether or not values outside of the intervals are part of the support.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        start: Union[int, float, List[float], np.ndarray],
        end: Union[int, float, List[float], np.ndarray],
        support_outside: Union[bool, List[bool], np.ndarray] = True,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``UniformLayer`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            start:
                Floating point, list of floats or one-dimensional NumPy array containing the start of the intervals (including).
                If a single floating point value is given, it is broadcast to all nodes.
            end:
                Floating point, list of floats or one-dimensional NumPy array containing the end of the intervals (including). Must be larger than 'start'.
                If a single floating point value is given, it is broadcast to all nodes.
            support_outside:
                Boolean, list of booleans or one-dimensional NumPy array containing booleans indicating whether or not values outside of the intervals are part of the support.
                If a single boolean value is given, it is broadcast to all nodes.
                Defaults to True.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'UniformLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'UniformLayer' was empty.")

            self._n_out = len(scope)

        super().__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [Uniform(s, 0.0, 1.0) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(start, end, support_outside)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def start(self) -> np.ndarray:
        """Returns the starts of the intervals of the represented distributions."""
        return np.array([node.start for node in self.nodes])

    @property
    def end(self) -> np.ndarray:
        """Returns the ends of the intervals of the represented distributions."""
        return np.array([node.end for node in self.nodes])

    @classmethod
    def accepts(self, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``UniformLayer`` can represent one or more univariate nodes with ``UniformType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not Uniform.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(
        self, signatures: List[FeatureContext]
    ) -> "UniformLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``UniformLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not self.accepts(signatures):
            raise ValueError(
                f"'UniformLayer' cannot be instantiated from the following signatures: {signatures}."
            )

        start = []
        end = []
        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if isinstance(domain, FeatureTypes.Uniform):
                start.append(domain.start)
                end.append(domain.end)
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'UniformLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return UniformLayer(scopes, start=start, end=end)

    @property
    def support_outside(self) -> np.ndarray:
        """Returns the booleans indicating whether or not values outside of the intervals are part of the supports of the represented distributions."""
        return np.array([node.support_outside for node in self.nodes])

    def set_params(
        self,
        start: Union[int, float, List[float], np.ndarray],
        end: Union[int, float, List[float], np.ndarray],
        support_outside: Union[bool, List[bool], np.ndarray] = True,
    ) -> None:
        """Sets the parameters for the represented distributions in the ``base`` backend.

        Args:
            start:
                Floating point, list of floats or one-dimensional NumPy array containing the start of the intervals (including).
                If a single floating point value is given, it is broadcast to all nodes.
            end:
                Floating point, list of floats or one-dimensional NumPy array containing the end of the intervals (including). Must be larger than 'start'.
                If a single floating point value is given, it is broadcast to all nodes.
            support_outside:
                Boolean, list of booleans or one-dimensional NumPy array containing booleans indicating whether or not values outside of the intervals are part of the support.
                If a single boolean value is given, it is broadcast to all nodes.
                Defaults to True.
        """
        if isinstance(start, int) or isinstance(start, float):
            start = np.array([float(start) for _ in range(self.n_out)])
        if isinstance(start, list):
            start = np.array(start)
        if start.ndim != 1:
            raise ValueError(
                f"Numpy array of start values for 'UniformLayer' is expected to be one-dimensional, but is {start.ndim}-dimensional."
            )
        if start.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of start values for 'UniformLayer' must match number of output nodes {self.n_out}, but is {start.shape[0]}"
            )

        if isinstance(end, int) or isinstance(end, float):
            end = np.array([float(end) for _ in range(self.n_out)])
        if isinstance(end, list):
            end = np.array(end)
        if end.ndim != 1:
            raise ValueError(
                f"Numpy array of end values for 'UniformLayer' is expected to be one-dimensional, but is {end.ndim}-dimensional."
            )
        if end.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of end values for 'UniformLayer' must match number of output nodes {self.n_out}, but is {end.shape[0]}"
            )

        if isinstance(support_outside, bool):
            support_outside = np.array(
                [support_outside for _ in range(self.n_out)]
            )
        if isinstance(support_outside, list):
            support_outside = np.array(support_outside)
        if support_outside.ndim != 1:
            raise ValueError(
                f"Numpy array of 'support_outside' values for 'UniformLayer' is expected to be one-dimensional, but is {support_outside.ndim}-dimensional."
            )
        if support_outside.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'support_outside' values for 'UniformLayer' must match number of output nodes {self.n_out}, but is {support_outside.shape[0]}"
            )

        for node_start, node_end, node_support_outside, node in zip(
            start, end, support_outside, self.nodes
        ):
            node.set_params(node_start, node_end, node_support_outside)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of three one-dimensional NumPy arrays representing the starts and ends of the intervals and the booleans indicating whether or not values outside of the intervals are part of the supports.
        """
        return self.start, self.end, self.support_outside

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

        Determines whether or note instances are part of the supports of the Uniform distributions, which are:

        .. math::

            \text{supp}(\text{Uniform})=\begin{cases} [start,end] & \text{if support\_outside}=\text{false}\\
                                                 (-\infty,\infty) & \text{if support\_outside}=\text{true} \end{cases}
        where
            - :math:`start` is the start of the interval
            - :math:`end` is the end of the interval
            - :math:`\text{support\_outside}` is a truth value indicating whether values outside of the interval are part of the support

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
    layer: UniformLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[UniformLayer, Uniform, None]:
    """Structural marginalization for ``UniformLayer`` objects.

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
        new_node = Uniform(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = UniformLayer(
            marg_scopes, *[np.array(p) for p in zip(*marg_params)]
        )
        return new_layer
