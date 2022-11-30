"""Contains Binomial leaf layer for SPFlow in the ``torch`` backend.
"""
from functools import reduce
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as D

from spflow.base.structure.general.layers.leaves.parametric.cond_binomial import (
    CondBinomialLayer as BaseCondBinomialLayer,
)
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaves.parametric.cond_binomial import (
    CondBinomial,
)
from spflow.torch.structure.module import Module


class CondBinomialLayer(Module):
    r"""Layer of multiple conditional (univariate) Binomial distribution leaf nodes in the ``torch`` backend.

    Represents multiple conditional univariate Binomial distributions with independent scopes, each with the following probability mass function (PMF):

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
        cond_f:
            Optional callable or list of callables to retrieve parameters for the leaf nodes.
            If a single callable, its output should be a dictionary containing ``p`` as a key, and the value should be
            a floating point, a list of floats or o one-dimensional NumPy array or PyTorch tensor, containing the success probabilities between zero and one.
            If it is a single floating point value, the same value is reused for all leaf nodes.
            If a list of callables, each one should return a dictionary containing``'p`` as a key, and the value should
            be a floating point value between zero and one.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        n: Union[int, List[int], np.ndarray, torch.Tensor],
        cond_f: Optional[Union[Callable, List[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``CondBinomialLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            n:
                Integer, list of integers or one-dimensional NumPy array containing the number of i.i.d. Bernoulli trials (greater or equal to 0) for each independent Binomial distribution.
                If a single integer value is given it is broadcast to all nodes.
            cond_f:
                Optional callable or list of callables to retrieve parameters for the leaf nodes.
                If a single callable, its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point, a list of floats or o one-dimensional NumPy array or PyTorch tensor, containing the success probabilities between zero and one.
                If it is a single floating point value, the same value is reused for all leaf nodes.
                If a list of callables, each one should return a dictionary containing ``p`` as a key, and the value should
                be a floating point value between zero and one.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.

        Raises:
            ValueError: Invalid arguments.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'CondBinomialLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'CondBinomialLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")
            if len(s.evidence) == 0:
                raise ValueError(f"Evidence scope for 'CondBinomialLayer' should not be empty.")

        super().__init__(children=[], **kwargs)

        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.empty(size=[]))

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.join(s2), self.scopes_out)

        # parse weights
        self.set_params(n)

        self.set_cond_f(cond_f)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CondBinomialLayer`` can represent one or more univariate nodes with ``BinomialType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not CondBinomial.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "CondBinomialLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``CondBinomialLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'CondBinomialLayer' cannot be instantiated from the following signatures: {signatures}.")

        n = []
        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if isinstance(domain, FeatureTypes.Binomial):
                n.append(domain.n)
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'CondBinomialLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return CondBinomialLayer(scopes, n=n)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    def set_cond_f(self, cond_f: Optional[Union[List[Callable], Callable]] = None) -> None:
        r"""Sets the ``cond_f`` property.

        Args:
            cond_f:
                Optional callable or list of callables to retrieve parameters for the leaf nodes.
                If a single callable, its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point, a list of floats or o one-dimensional NumPy array or PyTorch tensor, containing the success probabilities between zero and one.
                If it is a single floating point value, the same value is reused for all leaf nodes.
                If a list of callables, each one should return a dictionary containing ``p`` as a key, and the value should
                be a floating point value between zero and one.

        Raises:
            ValueError: If list of callables does not match number of nodes represented by the layer.
        """
        if isinstance(cond_f, List) and len(cond_f) != self.n_out:
            raise ValueError(
                "'CondBinomialLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes."
            )

        self.cond_f = cond_f

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> torch.Tensor:
        r"""Retrieves the conditional parameters of the leaf layer.

        First, checks if conditional parameter (``p``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function or list of functions (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameter.

        Args:
            data:
                Two-dimensional NumPy array containing the data to compute the conditional parameters.
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
            raise ValueError("'CondBinomialLayer' requires either 'p' or 'cond_f' to retrieve 'p' to be specified.")

        # if 'p' was not already specified, retrieve it
        if p is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, List):
                p = torch.tensor([f(data)["p"] for f in cond_f])
            else:
                p = cond_f(data)["p"]

        if isinstance(p, float) or isinstance(p, int):
            p = torch.tensor([p for _ in range(self.n_out)])
        elif isinstance(p, list) or isinstance(p, np.ndarray):
            p = torch.tensor(p)
        if p.ndim != 1:
            raise ValueError(
                f"Numpy array of 'p' values for 'CondBinomialLayer' is expected to be one-dimensional, but is {p.ndim}-dimensional."
            )
        if p.shape[0] == 1:
            p = torch.hstack([p for _ in range(self.n_out)])
        if p.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'p' values for 'CondBinomialLayer' must match number of output nodes {self.n_out}, but is {p.shape[0]}"
            )
        if torch.any(p < 0.0) or torch.any(p > 1.0) or not all(torch.isfinite(p)):
            raise ValueError(
                f"Values of 'p' for 'CondBinomialLayer' distribution must to be between 0.0 and 1.0, but are: {p}"
            )

        return p

    def dist(self, p: torch.Tensor, node_ids: Optional[List[int]] = None) -> D.Distribution:
        r"""Returns the PyTorch distributions represented by the leaf layer.

        Args:
            p:
                One-dimensional PyTorch tensor representing the success probabilities of all distributions between zero and one  (not just the ones specified by ``node_ids``).
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            ``torch.distributions.Binomial`` instance.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Binomial(total_count=self.n[node_ids], probs=p[node_ids])

    def set_params(self, n: Union[int, List[int], np.ndarray, torch.Tensor]) -> None:
        """Sets the parameters for the represented distributions.

        Args:
            n:
                Integer, list of integers or one-dimensional NumPy array or PyTorch tensor containing the number of i.i.d. Bernoulli trials (greater or equal to 0) for each independent Binomial distribution.
                If a single integer value is given it is broadcast to all nodes.
        """
        if isinstance(n, int) or isinstance(n, float):
            n = torch.tensor([n for _ in range(self.n_out)])
        elif isinstance(n, list) or isinstance(n, np.ndarray):
            n = torch.tensor(n)
        if n.ndim != 1:
            raise ValueError(
                f"Numpy array of 'n' values for 'BinomialLayer' is expected to be one-dimensional, but is {n.ndim}-dimensional."
            )
        if n.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'n' values for 'BinomialLayer' must match number of output nodes {self.n_out}, but is {n.shape[0]}"
            )

        if torch.any(n < 0) or not torch.any(torch.isfinite(n)):
            raise ValueError(f"Values for 'n' of 'BinomialLayer' must to greater of equal to 0, but was: {n}")

        if not torch.all(torch.remainder(n, 1.0) == torch.tensor(0.0)):
            raise ValueError(f"Values for 'n' of 'BinomialLayer' must be (equal to) an integer value, but was: {n}")

        node_scopes = torch.tensor([s.query[0] for s in self.scopes_out])

        for node_scope in torch.unique(node_scopes):
            # at least one such element exists
            n_values = n[node_scopes == node_scope]
            if not torch.all(n_values == n_values[0]):
                raise ValueError("All values of 'n' for 'BinomialLayer' over the same scope must be identical.")

        self.n.data = n

    def get_params(self) -> Tuple[torch.Tensor]:
        """Returns the parameters of the represented distribution.

        Returns:
            One-dimensional PyTorch tensor representing the number of i.i.d. Bernoulli trials.
        """
        return (self.n,)

    def check_support(
        self,
        data: torch.Tensor,
        node_ids: Optional[List[int]] = None,
        is_scope_data: bool = False,
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Binomial distributions, which are:

        .. math::

            \text{supp}(\text{Binomial})=\{0,\hdots,n\}

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
        valid = self.dist(torch.ones(self.n_out), node_ids).support.check(scope_data)  # type: ignore

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # set nan_entries back to True
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf()

        return valid


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: CondBinomialLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[CondBinomialLayer, CondBinomial, None]:
    """Structural marginalization for ``CondBinomialLayer`` objects in the ``torch`` backend.

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
        node_id = marginalized_node_ids.pop()
        return CondBinomial(scope=marginalized_scopes[0], n=layer.n[node_id].item())
    else:
        return CondBinomialLayer(scope=marginalized_scopes, n=layer.n[marginalized_node_ids].detach())


@dispatch(memoize=True)  # type: ignore
def toTorch(layer: BaseCondBinomialLayer, dispatch_ctx: Optional[DispatchContext] = None) -> CondBinomialLayer:
    """Conversion for ``CondBinomialLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondBinomialLayer(scope=layer.scopes_out, n=layer.n)


@dispatch(memoize=True)  # type: ignore
def toBase(
    torch_layer: CondBinomialLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> BaseCondBinomialLayer:
    """Conversion for ``CondBinomialLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondBinomialLayer(scope=torch_layer.scopes_out, n=torch_layer.n.numpy())
