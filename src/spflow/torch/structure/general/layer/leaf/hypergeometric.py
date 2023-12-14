"""Contains Hypergeometric leaf layer for SPFlow in the ``torch`` backend.
"""
from functools import reduce
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from spflow.base.structure.general.layer.leaf.hypergeometric import (
    HypergeometricLayer as BaseHypergeometricLayer,
)
from spflow.tensorly.structure.general.layer.leaf.general_hypergeometric import HypergeometricLayer as GeneralHypergeometricLayer
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.node.leaf.hypergeometric import (
    Hypergeometric,
)
from spflow.tensorly.structure.module import Module


class HypergeometricLayer(Module):
    r"""Layer of multiple (univariate) Hypergeometric distribution leaf node in the ``torch`` backend.

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
            One-dimensional PyTorch tensor specifying the total numbers of entities (in the populations), greater or equal to 0.
        M:
            One-dimensional PyTorch tensor specifying the numbers of entities with property of interest (in the populations), greater or equal to zero and less than or equal to N.
        n:
            One-dimensional PyTorch tensor specifying the numbers of draws, greater of equal to zero and less than or equal to N.
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
        """Initializes ``HypergeometricLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            N:
                Integer, list of ints or one-dimensional NumPy array or PyTorch tensor specifying the total numbers of entities (in the populations), greater or equal to 0.
                If a single value is given it is broadcast to all nodes.
            M:
                Integer, list of ints or one-dimensional NumPy array or PyTorch tensor specifying the numbers of entities with property of interest (in the populations), greater or equal to zero and less than or equal to N.
                If a single value is given it is broadcast to all nodes.
            n:
                Integer, list of ints or one-dimensional NumPy array or PyTorch tensor specifying the numbers of draws, greater of equal to zero and less than or equal to N.
                If a single value is given it is broadcast to all nodes.
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
                raise ValueError("List of scopes for 'HypergeometricLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")
            if len(s.evidence) != 0:
                raise ValueError(f"Evidence scope for 'HypergeometricLayer' should be empty, but was {s.evidence}.")

        super().__init__(children=[], **kwargs)

        # register number of trials n as torch buffer (should not be changed)
        #self.register_buffer("N", torch.empty(size=[]))
        #self.register_buffer("M", torch.empty(size=[]))
        #self.register_buffer("n", torch.empty(size=[]))
        self.N = torch.empty(size=[], device=self.device)
        self.M = torch.empty(size=[], device=self.device)
        self.n = torch.empty(size=[], device=self.device)

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.join(s2), self.scopes_out)

        # parse weights
        self.set_params(N, M, n)
        self.backend = "pytorch"

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``HypergeometricLayer`` can represent one or more univariate nodes with ``HypergeometricType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not Hypergeometric.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "HypergeometricLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``HypergeometricLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'HypergeometricLayer' cannot be instantiated from the following signatures: {signatures}."
            )

        N = []
        M = []
        n = []
        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if isinstance(domain, FeatureTypes.Hypergeometric):
                N.append(domain.N)
                M.append(domain.M)
                n.append(domain.n)
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'HypergeometricLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return HypergeometricLayer(scopes, N=N, M=M, n=n)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    def set_params(
        self,
        N: Union[int, List[int], np.ndarray, torch.Tensor],
        M: Union[int, List[int], np.ndarray, torch.Tensor],
        n: Union[int, List[int], np.ndarray, torch.Tensor],
    ) -> None:
        """Sets the parameters for the represented distributions.

        Args:
            N:
                Integer, list of ints or one-dimensional NumPy array or PyTorch tensor specifying the total numbers of entities (in the populations), greater or equal to 0.
                If a single value is given it is broadcast to all nodes.
            M:
                Integer, list of ints or one-dimensional NumPy array or PyTorch tensor specifying the numbers of entities with property of interest (in the populations), greater or equal to zero and less than or equal to N.
                If a single value is given it is broadcast to all nodes.
            n:
                Integer, list of ints or one-dimensional NumPy array or PyTorch tensor specifying the numbers of draws, greater of equal to zero and less than or equal to N.
                If a single value is given it is broadcast to all nodes.
        """
        if isinstance(N, int) or isinstance(N, float):
            N = torch.tensor([N for _ in range(self.n_out)], device=self.device)
        elif isinstance(N, list) or isinstance(N, np.ndarray):
            N = torch.tensor(N, device=self.device)
        if N.ndim != 1:
            raise ValueError(
                f"Torch tensor of 'N' values for 'HypergeometricLayer' is expected to be one-dimensional, but is {N.ndim}-dimensional."
            )
        if N.shape[0] != self.n_out:
            raise ValueError(
                f"Length of torch tensor of 'N' values for 'HypergeometricLayer' must match number of output nodes {self.n_out}, but is {N.shape[0]}"
            )

        if isinstance(M, int) or isinstance(M, float):
            M = torch.tensor([M for _ in range(self.n_out)], device=self.device)
        elif isinstance(n, list) or isinstance(M, np.ndarray):
            M = torch.tensor(M, device=self.device)
        if M.ndim != 1:
            raise ValueError(
                f"Torch tensor of 'M' values for 'HypergeometricLayer' is expected to be one-dimensional, but is {M.ndim}-dimensional."
            )
        if M.shape[0] != self.n_out:
            raise ValueError(
                f"Length of torch tensor of 'M' values for 'HypergeometricLayer' must match number of output nodes {self.n_out}, but is {M.shape[0]}"
            )

        if isinstance(n, int) or isinstance(n, float):
            n = torch.tensor([n for _ in range(self.n_out)], device=self.device)
        elif isinstance(n, list) or isinstance(n, np.ndarray):
            n = torch.tensor(n, device=self.device)
        if n.ndim != 1:
            raise ValueError(
                f"Torch tensor of 'n' values for 'HypergeometricLayer' is expected to be one-dimensional, but is {n.ndim}-dimensional."
            )
        if n.shape[0] != self.n_out:
            raise ValueError(
                f"Length of torch tensor of 'n' values for 'HypergeometricLayer' must match number of output nodes {self.n_out}, but is {n.shape[0]}"
            )

        if torch.any(N < 0) or not torch.all(torch.isfinite(N)):
            raise ValueError(f"Value of 'N' for 'HypergeometricLayer' must be greater of equal to 0, but was: {N}")
        if not torch.all(torch.remainder(N, 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Value of 'N' for 'HypergeometricLayer' must be (equal to) an integer value, but was: {N}"
            )

        if torch.any(M < 0) or torch.any(M > N) or not torch.all(torch.isfinite(M)):
            raise ValueError(
                f"Values of 'M' for 'HypergeometricLayer' must be greater of equal to 0 and less or equal to 'N', but was: {M}"
            )
        if not torch.all(torch.remainder(M, 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Values of 'M' for 'HypergeometricLayer' must be (equal to) an integer value, but was: {M}"
            )

        if torch.any(n < 0) or torch.any(n > N) or not torch.all(torch.isfinite(n)):
            raise ValueError(
                f"Value of 'n' for 'HypergeometricLayer' must be greater of equal to 0 and less or equal to 'N', but was: {n}"
            )
        if not torch.all(torch.remainder(n, 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Value of 'n' for 'HypergeometricLayer' must be (equal to) an integer value, but was: {n}"
            )

        node_scopes = torch.tensor([s.query[0] for s in self.scopes_out])

        for node_scope in torch.unique(node_scopes):
            # at least one such element exists
            N_values = N[node_scopes == node_scope]
            if not torch.all(N_values == N_values[0]):
                raise ValueError("All values of 'N' for 'HypergeometricLayer' over the same scope must be identical.")
            # at least one such element exists
            M_values = M[node_scopes == node_scope]
            if not torch.all(M_values == M_values[0]):
                raise ValueError("All values of 'M' for 'HypergeometricLayer' over the same scope must be identical.")
            # at least one such element exists
            n_values = n[node_scopes == node_scope]
            if not torch.all(n_values == n_values[0]):
                raise ValueError("All values of 'n' for 'HypergeometricLayer' over the same scope must be identical.")

        self.N.data = N.to(self.device)
        self.M.data = M.to(self.device)
        self.n.data = n.to(self.device)

    def get_trainable_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the parameters of the represented distribution.

        Returns:
            Thee one-dimensional PyTorch tensors representing the total numbers of entities, the numbers of entities of interest and the numbers of draws.
        """
        return self.N, self.M, self.n

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the parameters of the represented distribution.

        Returns:
            Thee one-dimensional PyTorch tensors representing the total numbers of entities, the numbers of entities of interest and the numbers of draws.
        """
        return self.N.cpu().detach().numpy(), self.M.cpu().detach().numpy(), self.n.cpu().detach().numpy()

    def check_support(
        self,
        data: torch.Tensor,
        node_ids: Optional[List[int]] = None,
        is_scope_data: bool = False,
    ) -> torch.Tensor:
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
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
                Assumes that relevant data is located in the columns corresponding to the scope indices.
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

        valid = torch.ones(scope_data.shape, dtype=torch.bool, device=self.device)

        # check for infinite values
        valid &= ~torch.isinf(scope_data)

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # check if all values are valid integers
        valid[~nan_mask] &= torch.remainder(scope_data[~nan_mask], 1) == 0

        node_ids_tensor = torch.tensor(node_ids)
        N_nodes = self.N[node_ids_tensor]
        M_nodes = self.M[node_ids_tensor]
        n_nodes = self.n[node_ids_tensor]

        # check if values are in valid range
        valid[~nan_mask & valid] &= (
            (
                scope_data
                >= torch.max(torch.vstack([torch.zeros(scope_data.shape[1], dtype=self.dtype, device=self.device), n_nodes + M_nodes - N_nodes,]), dim=0,)[
                    0
                ].unsqueeze(0)
            )
            & (  # type: ignore
                scope_data <= torch.min(torch.vstack([n_nodes, M_nodes]), dim=0)[0].unsqueeze(0)  # type: ignore
            )
        )[~nan_mask & valid]

        return valid

    def log_prob(self, k: torch.Tensor, node_ids: Optional[List[int]] = None) -> torch.Tensor:
        """Computes the log-likelihood for specified input data.

        The log-likelihoods of the Hypergeometric distribution are computed according to the logarithm of its probability mass function (PMF).

        Args:
            k:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            Two-dimensional PyTorch tensor containing the log-likelihoods of the corresponding input samples and nodes.
            Each row corresponds to an input sample.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        node_ids_tensor = torch.tensor(node_ids)

        N = self.N[node_ids_tensor]
        M = self.M[node_ids_tensor]
        n = self.n[node_ids_tensor]

        N_minus_M = N - M  # type: ignore
        n_minus_k = n - k  # type: ignore

        # ----- (M over m) * (N-M over n-k) / (N over n) -----

        # log_M_over_k = torch.lgamma(self.M+1) - torch.lgamma(self.M-k+1) - torch.lgamma(k+1)
        # log_NM_over_nk = torch.lgamma(N_minus_M+1) - torch.lgamma(N_minus_M-n_minus_k+1) - torch.lgamma(n_minus_k+1)
        # log_N_over_n = torch.lgamma(self.N+1) - torch.lgamma(self.N-self.n+1) - torch.lgamma(self.n+1)
        # result = log_M_over_k + log_NM_over_nk - log_N_over_n

        # ---- alternatively (more precise according to SciPy) -----
        # betaln(good+1, 1) + betaln(bad+1,1) + betaln(total-draws+1, draws+1) - betaln(k+1, good-k+1) - betaln(draws-k+1, bad-draws+k+1) - betaln(total+1, 1)

        lgamma_1 = torch.lgamma(torch.ones(len(node_ids), dtype=self.dtype, device=self.device))
        lgamma_M_p_2 = torch.lgamma(M + 2)
        lgamma_N_p_2 = torch.lgamma(N + 2)
        lgamma_N_m_M_p_2 = torch.lgamma(N_minus_M + 2)

        result = (
            torch.lgamma(M + 1)  # type: ignore
            + lgamma_1
            - lgamma_M_p_2  # type: ignore
            + torch.lgamma(N_minus_M + 1)  # type: ignore
            + lgamma_1
            - lgamma_N_m_M_p_2  # type: ignore
            + torch.lgamma(N - n + 1)  # type: ignore
            + torch.lgamma(n + 1)  # type: ignore
            - lgamma_N_p_2  # type: ignore
            - torch.lgamma(k + 1)  # .float()
            - torch.lgamma(M - k + 1)
            + lgamma_M_p_2  # type: ignore
            - torch.lgamma(n_minus_k + 1)
            - torch.lgamma(N_minus_M - n + k + 1)
            + lgamma_N_m_M_p_2  # type: ignore
            - torch.lgamma(N + 1)  # type: ignore
            - lgamma_1
            + lgamma_N_p_2  # type: ignore
        )

        return result

    def to_dtype(self, dtype):
        self.dtype = dtype

    def to_device(self, device):
        self.device = device
        self.set_params(self.N.data, self.M.data, self.n.data)



@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: HypergeometricLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[HypergeometricLayer, Hypergeometric, None]:
    """Structural marginalization for ``HypergeometricLayer`` objects in the ``torch`` backend.

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
        return Hypergeometric(
            scope=marginalized_scopes[0],
            N=layer.N[node_id].item(),
            M=layer.M[node_id].item(),
            n=layer.n[node_id].item(),
        )
    else:
        return HypergeometricLayer(
            scope=marginalized_scopes,
            N=layer.N[marginalized_node_ids],
            M=layer.M[marginalized_node_ids],
            n=layer.n[marginalized_node_ids],
        )


@dispatch(memoize=True)  # type: ignore
def toTorch(
    layer: BaseHypergeometricLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> HypergeometricLayer:
    """Conversion for ``HypergeometricLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return HypergeometricLayer(scope=layer.scopes_out, N=layer.N, M=layer.M, n=layer.n)


@dispatch(memoize=True)  # type: ignore
def toBase(layer: HypergeometricLayer, dispatch_ctx: Optional[DispatchContext] = None) -> BaseHypergeometricLayer:
    """Conversion for ``HypergeometricLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseHypergeometricLayer(
        scope=layer.scopes_out,
        N=layer.N.numpy(),
        M=layer.M.numpy(),
        n=layer.n.numpy(),
    )

@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: HypergeometricLayer, dispatch_ctx: Optional[DispatchContext] = None):
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return GeneralHypergeometricLayer( scope=leaf_node.scopes_out,
        N=leaf_node.N.detach().numpy(),
        M=leaf_node.M.detach().numpy(),
        n=leaf_node.n.detach().numpy())