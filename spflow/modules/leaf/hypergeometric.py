import torch
from torch import Tensor
from typing import Optional, Callable

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule, MLEBatch
from spflow.utils.leaf import parse_leaf_args
from spflow.utils.cache import Cache


class Hypergeometric(LeafModule):
    # Parameters are registered buffers (fixed), so descriptor refactor does not apply.
    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        K: Tensor = None,
        N: Tensor = None,
        n: Tensor = None,
    ):
        r"""
        Initialize a Hypergeometric distribution leaf module.

        Args:
            scope: Scope object specifying the scope of the distribution.
            out_channels: The number of output channels. If None, it is determined by the parameter tensors.
            num_repetitions: The number of repetitions for the leaf module.
            K: PyTorch tensor specifying the total numbers of entities (in the populations), greater or equal to 0.
            N: PyTorch tensor specifying the numbers of entities with property of interest (in the populations), greater or equal to zero and less than or equal to N.
            n: PyTorch tensor specifying the numbers of draws, greater of equal to zero and less than or equal to N.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[K, N, n], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        self.register_buffer("K", torch.empty(size=[]))
        self.register_buffer("N", torch.empty(size=[]))
        self.register_buffer("n", torch.empty(size=[]))

        self.check_inputs(K, N, n, event_shape)

        self.K = K
        self.N = N
        self.n = n

    def check_inputs(self, K: Tensor, N: Tensor, n: Tensor, event_shape: tuple[int, ...]):
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
        if len(event_shape) > 1:
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
        return _HypergeometricDistribution(self.K, self.N, self.n, self.K.shape)

    @property
    def _supported_value(self):
        return self.n + self.K - self.N

    def _mle_compute_statistics(self, batch: MLEBatch) -> dict[str, Tensor]:
        """Hypergeometric parameters are fixed buffers; nothing to estimate."""
        return {}

    def params(self) -> dict[str, Tensor]:
        return {"K": self.K, "N": self.N, "n": self.n}

    def _custom_support_mask(self, data: Tensor) -> Tensor:
        """Hypergeometric support requires integer counts within valid bounds."""
        original_data_shape = data.shape
        K = self.K.to(dtype=data.dtype, device=data.device)
        N = self.N.to(dtype=data.dtype, device=data.device)
        n_param = self.n.to(dtype=data.dtype, device=data.device)

        min_successes = torch.maximum(
            torch.zeros_like(N),
            n_param + K - N,
        )
        max_successes = torch.minimum(n_param, K)

        # Add batch dimension to parameters
        min_successes_expanded = min_successes.unsqueeze(
            0
        )  # (1, features, channels) or (1, features, channels, repetitions)
        max_successes_expanded = max_successes.unsqueeze(0)

        # If data has fewer dimensions than min_successes, expand it for the comparison
        # This handles the case where data is (batch, features) but min_successes is (1, features, channels)
        data_expanded = data
        num_added_dims = 0
        while data_expanded.dim() < min_successes_expanded.dim():
            data_expanded = data_expanded.unsqueeze(-1)  # Add trailing dimensions
            num_added_dims += 1

        integer_mask = torch.remainder(data_expanded, 1) == 0
        range_mask = (data_expanded >= min_successes_expanded) & (data_expanded <= max_successes_expanded)
        mask = integer_mask & range_mask

        # Reduce the mask back to the original data shape
        # After broadcasting, the mask includes extra dimensions, so we take the first element along those
        for _ in range(num_added_dims):
            mask = mask[..., 0]  # Select first element along last dimension

        return mask


class _HypergeometricDistribution:
    def __init__(self, K: torch.Tensor, N: torch.Tensor, n: torch.Tensor, event_shape: tuple[int, ...]):
        # super(_HypergeometricDistribution, self).__init__(event_shape=K.shape)
        self.N = N
        self.K = K
        self.n = n
        self.event_shape = event_shape
        self.batch_shape = ()  # No batch shape for hypergeometric, event_shape contains all the structure

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
        rand_indices = torch.argsort(
            torch.rand(*sample_shape, self.N.max().to(torch.int32).item(), device=self.K.device), dim=-1
        )

        # Use broadcasting to create masks where draws are of interest
        K_expanded = self.K.unsqueeze(0).expand(*n_samples, *self.K.shape)
        n_expanded = self.n.unsqueeze(0).expand(*n_samples, *self.n.shape)

        # Create a mask for the "drawn" indices, considering the first K indices as objects of interest
        drawn_mask = rand_indices < K_expanded.unsqueeze(-1)

        # Count the "drawn" indices for each sample, within the first 'n' draws
        n_drawn = drawn_mask[..., : n_expanded.max().to(torch.int32).item()].sum(dim=-1)

        # Adjust the shape of n_drawn to match the desired sample shape
        n_drawn_shape_adjusted = n_drawn[..., : self.n.shape[-1]]

        # Ensure the counts do not exceed the limits defined by n and K for each scope and leaf
        data = torch.where(n_drawn_shape_adjusted < n_expanded, n_drawn_shape_adjusted, n_expanded)

        return data
