import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule


class _HypergeometricDistribution:
    """Custom Hypergeometric distribution implementation.

    Since PyTorch doesn't have a built-in Hypergeometric distribution,
    this class implements the necessary methods for inference and sampling.
    """

    def __init__(self, K: torch.Tensor, N: torch.Tensor, n: torch.Tensor, validate_args: bool = True):
        self.N = N
        self.K = K
        self.n = n
        self.event_shape = n.shape
        self.batch_shape = ()  # No batch shape for hypergeometric, event_shape contains all the structure
        self.validate_args = validate_args

    def _align_support_mask(self, mask: Tensor, data: Tensor) -> Tensor:
        """Align a support mask with the shape of ``data`` for boolean indexing."""
        if mask.dim() < data.dim():
            expand_dims = data.dim() - mask.dim()
            mask = mask.reshape(*mask.shape, *([1] * expand_dims))

        if mask.dim() != data.dim():
            raise RuntimeError(
                f"Support mask rank {mask.dim()} incompatible with data rank {data.dim()} "
                f"in {self.__class__.__name__}. Provide a custom check_support override."
            )

        slices: list[slice] = []
        for mask_size, data_size in zip(mask.shape, data.shape):
            if mask_size == data_size:
                slices.append(slice(None))
            elif data_size == 1 and mask_size > 1:
                slices.append(slice(0, 1))
            elif mask_size == 1 and data_size > 1:
                slices.append(slice(None))
            else:
                raise RuntimeError(
                    f"Support mask shape {tuple(mask.shape)} incompatible with data shape "
                    f"{tuple(data.shape)} in {self.__class__.__name__}."
                )

        mask = mask[tuple(slices)]
        if mask.shape != data.shape:
            mask = mask.expand_as(data)
        return mask

    def check_support(self, data: Tensor) -> Tensor:
        """Hypergeometric support: integer counts within valid bounds.

        Valid range: max(0, n+K-N) <= x <= min(n, K)
        """
        nan_mask = torch.isnan(data)
        valid = torch.ones_like(data, dtype=torch.bool)

        K = self.K.to(dtype=data.dtype, device=data.device)
        N = self.N.to(dtype=data.dtype, device=data.device)
        n_param = self.n.to(dtype=data.dtype, device=data.device)

        min_successes = torch.maximum(torch.zeros_like(N), n_param + K - N)
        max_successes = torch.minimum(n_param, K)

        # Add batch dimensions to parameters
        min_successes_expanded = min_successes.unsqueeze(0)
        max_successes_expanded = max_successes.unsqueeze(0)

        # Expand data to match parameter dimensions if needed
        data_expanded = data
        num_added_dims = 0
        while data_expanded.dim() < min_successes_expanded.dim():
            data_expanded = data_expanded.unsqueeze(-1)
            num_added_dims += 1

        integer_mask = torch.remainder(data_expanded, 1) == 0
        range_mask = (data_expanded >= min_successes_expanded) & (data_expanded <= max_successes_expanded)
        support_mask = integer_mask & range_mask

        # Reduce the mask back to original data shape
        for _ in range(num_added_dims):
            support_mask = support_mask[..., 0]

        # Apply support mask
        aligned = self._align_support_mask(support_mask, data)
        valid[~nan_mask] &= aligned[~nan_mask]

        # Reject infinities
        valid[~nan_mask & valid] &= ~data[~nan_mask & valid].isinf()

        invalid_mask = (~valid) & (~nan_mask)
        if invalid_mask.any():
            invalid_values = data[invalid_mask].detach().cpu().tolist()
            raise ValueError(f"Hypergeometric received data outside of support: {invalid_values}")

        return valid

    @property
    def mode(self):
        """Return the mode of the distribution."""
        return torch.floor((self.n + 1) * (self.K + 1) / (self.N + 2))

    def log_prob(self, k: torch.Tensor) -> torch.Tensor:
        """Compute log probability using logarithmic identities to avoid overflow."""
        N = self.N
        K = self.K
        n = self.n
        if self.validate_args:
            support_mask = self.check_support(k)

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

        result = result.masked_fill(~support_mask, float("-inf"))

        return result

    def sample(self, n_samples):
        """Efficiently samples from the hypergeometric distribution in parallel for all scope_idx and leaf_idx.

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

        # Generate random indices for each sample, scope, and leaves
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

        # Ensure the counts do not exceed the limits defined by n and K for each scope and leaves
        data = torch.where(n_drawn_shape_adjusted < n_expanded, n_drawn_shape_adjusted, n_expanded)

        return data


class Hypergeometric(LeafModule):
    """Hypergeometric distribution leaf for sampling without replacement.

    All parameters (K, N, n) are fixed buffers and cannot be learned.

    Attributes:
        K: Number of success states in population (fixed buffer).
        N: Population size (fixed buffer).
        n: Number of draws (fixed buffer).
        distribution: Underlying custom Hypergeometric distribution.
    """

    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = 1,
        K: Tensor | None = None,
        N: Tensor | None = None,
        n: Tensor | None = None,
        validate_args: bool | None = True,
    ):
        """Initialize Hypergeometric distribution leaf module.

        Args:
            scope: Scope object specifying the scope of the distribution.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions for the distribution.
            K: Number of success states tensor.
            N: Population size tensor.
            n: Number of draws tensor.
            validate_args: Whether to enable argument validation.
        """
        if K is None or N is None or n is None:
            raise InvalidParameterCombinationError(
                "'K', 'N', and 'n' parameters are required for Hypergeometric distribution"
            )

        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[K, N, n],
            validate_args=validate_args,
        )

        # Register fixed buffers
        self.register_buffer("K", torch.empty(size=[]))
        self.register_buffer("N", torch.empty(size=[]))
        self.register_buffer("n", torch.empty(size=[]))

        # Validate inputs before assignment
        self.check_inputs(K, N, n, self.event_shape)

        # Assign parameters
        self.K = K
        self.N = N
        self.n = n

    def check_inputs(self, K: Tensor, N: Tensor, n: Tensor, event_shape: tuple[int, ...]):
        """Validate hypergeometric parameters."""
        if torch.any(N < 0) or not torch.all(torch.isfinite(N)):
            raise ValueError(f"Value of 'N' for 'Hypergeometric' must be greater of equal to 0, but was: {N}")
        if not torch.all(torch.remainder(N, 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Value of 'N' for 'Hypergeometric' must be (equal to) an integer value, but was: {N}"
            )

        if torch.any(K < 0) or torch.any(K > N) or not torch.all(torch.isfinite(K)):
            raise ValueError(
                f"Values of 'K' for 'Hypergeometric' must be greater of equal to 0 and less or equal to 'N', but was: {K}"
            )
        if not torch.all(torch.remainder(K, 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Values of 'K' for 'Hypergeometric' must be (equal to) an integer value, but was: {K}"
            )

        if torch.any(n < 0) or torch.any(n > N) or not torch.all(torch.isfinite(n)):
            raise ValueError(
                f"Value of 'n' for 'Hypergeometric' must be greater of equal to 0 and less or equal to 'N', but was: {n}"
            )
        if not torch.all(torch.remainder(n, 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Value of 'n' for 'Hypergeometric' must be (equal to) an integer value, but was: {n}"
            )
        if len(event_shape) > 1:
            if not (N == N[0]).all(dim=0).all():
                raise ValueError(
                    "All values of 'N' for 'Hypergeometric' over the same scope must be identical."
                )
            if not (K == K[0]).all(dim=0).all():
                raise ValueError(
                    "All values of 'K' for 'Hypergeometric' over the same scope must be identical."
                )
            if not (n == n[0]).all(dim=0).all():
                raise ValueError(
                    "All values of 'n' for 'Hypergeometric' over the same scope must be identical."
                )

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return self.n + self.K - self.N

    @property
    def _torch_distribution_class(self) -> type[_HypergeometricDistribution]:
        return _HypergeometricDistribution

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"K": self.K, "N": self.N, "n": self.n}

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> dict[str, Tensor]:
        """Compute raw MLE estimates for hypergeometric distribution (without broadcasting).

        Since hypergeometric parameters are fixed buffers and cannot be learned,
        this method returns the current parameter values unchanged.

        Args:
            data: Input data tensor.
            weights: Weight tensor for each data point.
            bias_correction: Not used for Hypergeometric (fixed parameters).

        Returns:
            Dictionary with 'K', 'N', and 'n' estimates (shape: out_features).
        """
        # Hypergeometric parameters are fixed - return current values
        return {"K": self.K, "N": self.N, "n": self.n}

    def _set_mle_parameters(self, params_dict: dict[str, Tensor]) -> None:
        """Set MLE-estimated parameters for Hypergeometric distribution.

        Since all parameters (K, N, n) are fixed buffers and cannot be learned,
        this method is a no-op. Hypergeometric parameters remain constant.

        Args:
            params_dict: Dictionary with 'K', 'N', and 'n' parameter values (unused).
        """
        # Hypergeometric parameters are fixed buffers - no updates needed
        pass

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Hypergeometric parameters are fixed buffers; nothing to estimate.

        Args:
            data: Scope-filtered data.
            weights: Normalized weights.
            bias_correction: Not used for Hypergeometric (fixed parameters).
        """
        # For consistency with the pattern, call _compute_parameter_estimates
        # but don't use the result since parameters are fixed
        _ = self._compute_parameter_estimates(data, weights, bias_correction)
        # No assignment needed - parameters are fixed buffers

    def check_support(self, data: Tensor) -> Tensor:
        """Check if data is in support of the Hypergeometric distribution.

        Delegates to the _HypergeometricDistribution's check_support method.

        Args:
            data: Input data tensor.

        Returns:
            Boolean tensor indicating which values are in support.
        """
        return self.distribution.check_support(data)
