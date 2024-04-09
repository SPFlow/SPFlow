#!/usr/bin/env python3

import torch
from torch import Tensor
import numpy as np

from spflow.distributions.distribution import Distribution
from spflow.meta.data import FeatureContext, FeatureTypes


class Hypergeometric(Distribution):
    def __init__(self, K: Tensor, N: Tensor, n: Tensor, event_shape: tuple[int, ...] = None):
        r"""Initializes ``Hypergeometric`` leaf node.

        Args:
            scope: Scope object specifying the scope of the distribution.
            K: PyTorch tensor specifying the total numbers of entities (in the populations), greater or equal to 0.
            N: PyTorch tensor specifying the numbers of entities with property of interest (in the populations), greater or equal to zero and less than or equal to N.
            n: PyTorch tensor specifying the numbers of draws, greater of equal to zero and less than or equal to N.
            n_out: Number of nodes per scope. Only relevant if mean and std is None.
        """
        if event_shape is None:
            event_shape = K.shape
        super().__init__(event_shape=event_shape)

        self.register_buffer("K", torch.empty(size=[]))
        self.register_buffer("N", torch.empty(size=[]))
        self.register_buffer("n", torch.empty(size=[]))

        self.check_inputs(K, N, n)

        self.K = K
        self.N = N
        self.n = n


    def check_inputs(self, K: Tensor, N: Tensor, n: Tensor):
        if torch.any(N < 0) or not torch.all(torch.isfinite(N)):
            raise ValueError(
                f"Value of 'N' for 'HypergeometricLayer' must be greater of equal to 0, but was: {N}"
            )
        if not torch.all(torch.remainder(N, 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Value of 'N' for 'HypergeometricLayer' must be (equal to) an integer value, but was: {N}"
            )

        if torch.any(K < 0) or torch.any(K > N) or not torch.all(torch.isfinite(K)):
            raise ValueError(
                f"Values of 'K' for 'HypergeometricLayer' must be greater of equal to 0 and less or equal to 'N', but was: {K}"
            )
        if not torch.all(torch.remainder(K, 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Values of 'K' for 'HypergeometricLayer' must be (equal to) an integer value, but was: {K}"
            )

        if torch.any(n < 0) or torch.any(n > N) or not torch.all(torch.isfinite(n)):
            raise ValueError(
                f"Value of 'n' for 'HypergeometricLayer' must be greater of equal to 0 and less or equal to 'N', but was: {n}"
            )
        if not torch.all(torch.remainder(n, 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Value of 'n' for 'HypergeometricLayer' must be (equal to) an integer value, but was: {n}"
            )
        if len(self.event_shape) > 1:
            if not (N == N[0]).all(dim=0).all():
                raise ValueError(
                    "All values of 'N' for 'HypergeometricLayer' over the same scope must be identical."
                )
            if not (K == K[0]).all(dim=0).all():
                raise ValueError(
                    "All values of 'N' for 'HypergeometricLayer' over the same scope must be identical."
                )
            if not (n == n[0]).all(dim=0).all():
                raise ValueError(
                    "All values of 'N' for 'HypergeometricLayer' over the same scope must be identical."
                )


    @property
    def distribution(self):
        return _HypergeometricDistribution(self.K, self.N, self.n, self.event_shape)



    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        feature_ctx = signatures[0]
        domains = feature_ctx.get_domains()

        # leaf is a single non-conditional univariate node
        if (
                len(domains) != 1
                or len(feature_ctx.scope.query) != len(domains)
                or len(feature_ctx.scope.evidence) != 0
        ):
            return False

        # leaf is a discrete Hypergeometric distribution
        # NOTE: only accept instances of 'FeatureTypes.Hypergeometric', otherwise required parameters 'N','K','n' are not specified. Reject 'FeatureTypes.Discrete' for the same reason.
        if not isinstance(domains[0], FeatureTypes.Hypergeometric):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Hypergeometric":
        if not cls.accepts(signatures):
            raise ValueError(
                f"'Hypergeometric' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if isinstance(domain, FeatureTypes.Hypergeometric):
            N, K, n = domain.N, domain.K, domain.n
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Hypergeometric' that was not caught during acception checking."
            )

        return Hypergeometric(N=N, K=K, n=n)


    def maximum_likelihood_estimation(self, data: Tensor, weights: Tensor = None, bias_correction=True):

        """
        All parameters of the Uniform distribution are regarded as fixed and will not be estimated.
        Therefore, this method does nothing, but check for the validity of the data.
        """
        data = data.unsqueeze(2)
        if torch.any(~ self.check_support(data)):
            raise ValueError("Encountered values outside of the support for uniform distribution.")

        # do nothing since there are no learnable parameters
        pass

    def marginalized_params(self, indices: list[int]) -> dict[str, Tensor]:
        return {"K": self.K[indices], "N": self.N[indices], "n": self.n[indices]}

    def check_support(
            self,
            data: torch.Tensor,
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.
        """

        valid = torch.ones(data.shape, dtype=torch.bool, device=data.device)

        # check for infinite values
        valid &= ~torch.isinf(data)

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(data)

        # check if all values are valid integers
        valid[~nan_mask] &= torch.remainder(data[~nan_mask], 1) == 0

        N_nodes = self.N
        K_nodes = self.K
        n_nodes = self.n

        # check if values are in valid range
        valid[~nan_mask & valid] &= (
                (
                        data
                        >= torch.max(
                    torch.vstack(
                        [
                            torch.zeros(self.event_shape, dtype=data.dtype, device=data.device),
                            n_nodes + K_nodes - N_nodes,
                        ]
                    ),
                    dim=0,
                )[0].unsqueeze(0)
                )
                & (  # type: ignore
                        data <= torch.min(torch.vstack([n_nodes, K_nodes]), dim=0)[0].unsqueeze(0)  # type: ignore
                )
        )[...,:1][~nan_mask & valid]

        return valid


class _HypergeometricDistribution():
    def __init__(self, K: torch.Tensor, N: torch.Tensor, n: torch.Tensor, event_shape: tuple[int, ...]):
        #super(_HypergeometricDistribution, self).__init__(event_shape=K.shape)
        self.N = N
        self.K = K
        self.n = n
        self.event_shape = event_shape

    @property
    def mode(self):
        return torch.floor((self.n + 1) * (self.K + 1) / (self.N + 2))

    def log_prob(self, k: torch.Tensor) -> torch.Tensor:
        N = self.N
        K = self.K
        n = self.n


        N_minus_K = N - K  # type: ignore
        n_minus_k = n - k  # type: ignore

        # ----- (K over m) * (N-K over n-k) / (N over n) -----

        lgamma_1 = torch.lgamma(torch.ones(self.event_shape, dtype=k.dtype, device=k.device))
        lgamma_K_p_2 = torch.lgamma(K + 2)
        lgamma_N_p_2 = torch.lgamma(N + 2)
        lgamma_N_m_K_p_2 = torch.lgamma(N_minus_K + 2)

        result = (
                torch.lgamma(K + 1)  # type: ignore
                + lgamma_1
                - lgamma_K_p_2  # type: ignore
                + torch.lgamma(N_minus_K + 1)  # type: ignore
                + lgamma_1
                - lgamma_N_m_K_p_2  # type: ignore
                + torch.lgamma(N - n + 1)  # type: ignore
                + torch.lgamma(n + 1)  # type: ignore
                - lgamma_N_p_2  # type: ignore
                - torch.lgamma(k + 1)  # .float()
                - torch.lgamma(K - k + 1)
                + lgamma_K_p_2  # type: ignore
                - torch.lgamma(n_minus_k + 1)
                - torch.lgamma(N_minus_K - n + k + 1)
                + lgamma_N_m_K_p_2  # type: ignore
                - torch.lgamma(N + 1)  # type: ignore
                - lgamma_1
                + lgamma_N_p_2  # type: ignore
        )

        return result


    def sample(self, n_samples):
        """
        Efficiently samples from the hypergeometric distribution in parallel for all scope_idx and leaf_idx.
        Args:
            n_samples (tuple): Number of samples to generate.

        Returns:
            torch.Tensor: Sampled values.
        """
        # Ensure n_samples is a tuple for consistency in operations
        if not isinstance(n_samples, tuple):
            n_samples = (n_samples,)

        # Prepare the tensor to store the samples
        sample_shape = n_samples + self.event_shape
        data = torch.zeros(sample_shape, device=self.K.device)

        # Generate random indices for each sample, scope, and leaf
        rand_indices = torch.argsort(torch.rand(*sample_shape, self.N.max().to(torch.int32).item(), device=self.K.device), dim=-1)

        # Use broadcasting to create masks where draws are of interest
        K_expanded = self.K.unsqueeze(0).expand(*n_samples, *self.K.shape)
        n_expanded = self.n.unsqueeze(0).expand(*n_samples, *self.n.shape)

        # Create a mask for the "drawn" indices, considering the first K indices as objects of interest
        drawn_mask = rand_indices < K_expanded.unsqueeze(-1)

        # Count the "drawn" indices for each sample, within the first 'n' draws
        n_drawn = drawn_mask[..., :n_expanded.max().to(torch.int32).item()].sum(dim=-1)

        # Adjust the shape of n_drawn to match the desired sample shape
        n_drawn_shape_adjusted = n_drawn[..., :self.n.shape[-1]]

        # Ensure the counts do not exceed the limits defined by n and K for each scope and leaf
        data = torch.where(n_drawn_shape_adjusted < n_expanded, n_drawn_shape_adjusted, n_expanded)

        return data
