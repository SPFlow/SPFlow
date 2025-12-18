"""Piecewise linear leaf distribution module.

This module provides a non-parametric density estimation approach that
approximates data distributions using piecewise linear functions constructed
from histograms. It uses K-means clustering to create multiple distributions
per leaf.
"""

from __future__ import annotations

import itertools
import logging
from typing import List, Optional

import torch
from torch import Tensor, nn

from spflow.meta.data.scope import Scope
from spflow.modules.leaves.leaf import LeafModule
from spflow.utils.cache import Cache
from spflow.utils.domain import DataType, Domain
from spflow.utils.histogram import get_bin_edges_torch
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context

logger = logging.getLogger(__name__)


def pairwise(iterable):
    """Iterate over consecutive pairs.

    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def interp(
    x: Tensor, xp: Tensor, fp: Tensor, dim: int = -1, extrapolate: str = "constant"
) -> Tensor:
    """One-dimensional linear interpolation between monotonically increasing sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points (xp, fp), evaluated at x.

    Source: https://github.com/pytorch/pytorch/issues/50334#issuecomment-2304751532

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
        xp: The x-coordinates of the data points, must be increasing.
        fp: The y-coordinates of the data points, same shape as xp.
        dim: Dimension across which to interpolate.
        extrapolate: How to handle values outside the range of xp. Options:
            - 'linear': Extrapolate linearly beyond range.
            - 'constant': Use boundary value of fp for x outside xp.

    Returns:
        The interpolated values, same size as x.
    """
    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1)
    xp = xp.movedim(dim, -1)
    fp = fp.movedim(dim, -1)

    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[..., :-1] - m * xp[..., :-1]  # offset
    
    # Ensure contiguous inputs for searchsorted
    xp = xp.contiguous()
    x = x.contiguous()
    indices = torch.searchsorted(xp, x, right=False)

    if extrapolate == "constant":
        # Pad m and b to get constant values outside of xp range
        m = torch.cat(
            [torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1
        )
        b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    else:  # extrapolate == 'linear'
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

    values = m.gather(-1, indices) * x + b.gather(-1, indices)

    values = values.clamp(min=0.0)

    return values.movedim(-1, dim)


class PiecewiseLinearDist:
    """Custom distribution for piecewise linear density estimation.

    Mimics the torch.distributions interface with log_prob, sample, and mode methods.

    Attributes:
        xs: Nested list of x-coordinates [R][L][F][C] where R=repetitions, L=leaves,
            F=features, C=channels.
        ys: Nested list of y-coordinates (densities) with same structure as xs.
        domains: List of Domain objects, one per feature.
    """

    def __init__(self, xs: List, ys: List, domains: List[Domain]):
        """Initialize the piecewise linear distribution.

        Args:
            xs: Nested list of x-coordinates for piecewise linear functions.
            ys: Nested list of y-coordinates (densities) for piecewise linear functions.
            domains: List of Domain objects describing each feature's domain.
        """
        self.xs = xs
        self.ys = ys
        self.domains = domains

        self.num_repetitions = len(xs)
        self.num_leaves = len(xs[0])
        self.num_features = len(xs[0][0])
        self.num_channels = len(xs[0][0][0])

    def _compute_cdf(self, xs: Tensor, ys: Tensor) -> Tensor:
        """Compute the CDF for the given piecewise linear function.

        Args:
            xs: X-coordinates of the piecewise function.
            ys: Y-coordinates (densities) of the piecewise function.

        Returns:
            CDF values at each x-coordinate.
        """
        # Compute the integral over each interval using the trapezoid rule
        intervals = torch.diff(xs)
        trapezoids = 0.5 * intervals * (ys[:-1] + ys[1:])  # Partial areas

        # Cumulative sum to build the CDF
        cdf = torch.cat([torch.zeros(1, device=xs.device), torch.cumsum(trapezoids, dim=0)])

        # Normalize the CDF to ensure it goes from 0 to 1
        cdf = cdf / (cdf[-1] + 1e-10)

        return cdf

    def sample(self, sample_shape: torch.Size | tuple[int, ...]) -> Tensor:
        """Sample from the piecewise linear distribution.

        Args:
            sample_shape: Shape of samples to generate.

        Returns:
            Samples tensor of shape (sample_shape[0], C, F, L, R).
        """
        num_samples = sample_shape[0] if isinstance(sample_shape, torch.Size) else sample_shape[0]
        samples = torch.empty(
            (
                num_samples,
                self.num_channels,
                self.num_features,
                self.num_leaves,
                self.num_repetitions,
            ),
            device=self.xs[0][0][0][0].device,
        )

        for i_feature in range(self.num_features):
            for i_channel in range(self.num_channels):
                for i_repetition in range(self.num_repetitions):
                    for i_leaf in range(self.num_leaves):
                        xs_i = self.xs[i_repetition][i_leaf][i_feature][i_channel]
                        ys_i = self.ys[i_repetition][i_leaf][i_feature][i_channel]

                        if self.domains[i_feature].data_type == DataType.DISCRETE:
                            # Sample from a categorical distribution
                            ys_i_wo_tails = ys_i[1:-1]  # Cut off the tail breaks
                            dist = torch.distributions.Categorical(probs=ys_i_wo_tails)
                            samples[
                                :, i_channel, i_feature, i_leaf, i_repetition
                            ] = dist.sample(sample_shape)
                        elif self.domains[i_feature].data_type == DataType.CONTINUOUS:
                            # Compute the CDF for this piecewise function
                            cdf = self._compute_cdf(xs_i, ys_i)

                            # Sample from a uniform distribution
                            u = torch.rand(num_samples, device=xs_i.device)

                            # Find the corresponding segment using searchsorted
                            # Ensure contiguous inputs
                            cdf = cdf.contiguous()
                            u = u.contiguous()
                            indices = torch.searchsorted(cdf, u, right=True)

                            # Clamp indices to be within valid range
                            indices = torch.clamp(indices, 1, len(xs_i) - 1)

                            # Perform linear interpolation to get the sample value
                            x0, x1 = xs_i[indices - 1], xs_i[indices]
                            cdf0, cdf1 = cdf[indices - 1], cdf[indices]
                            slope = (x1 - x0) / (cdf1 - cdf0 + 1e-8)  # Avoid division by zero

                            # Compute the sampled value
                            samples[:, i_channel, i_feature, i_leaf, i_repetition] = (
                                x0 + slope * (u - cdf0)
                            )
                        else:
                            raise ValueError(
                                f"Unknown data type: {self.domains[i_feature].data_type}"
                            )

        return samples

    @property
    def mode(self) -> Tensor:
        """Compute the mode of the distribution.

        Returns:
            Modes tensor of shape (C, F, L, R).
        """
        modes = torch.empty(
            (
                self.num_channels,
                self.num_features,
                self.num_leaves,
                self.num_repetitions,
            ),
            device=self.xs[0][0][0][0].device,
        )

        for i_feature in range(self.num_features):
            for i_channel in range(self.num_channels):
                for i_repetition in range(self.num_repetitions):
                    for i_leaf in range(self.num_leaves):
                        xs_i = self.xs[i_repetition][i_leaf][i_feature][i_channel]
                        ys_i = self.ys[i_repetition][i_leaf][i_feature][i_channel]

                        # Find the mode (the x value with the highest PDF value)
                        max_idx = torch.argmax(ys_i)
                        mode_value = xs_i[max_idx]

                        # Store the mode value
                        modes[i_channel, i_feature, i_leaf, i_repetition] = mode_value

        return modes

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probabilities for input data.

        Args:
            x: Input tensor of shape (N, C, F, 1, 1) or (N, C, F).

        Returns:
            Log probabilities of shape (N, C, F, L, R).
        """
        # Handle input shapes
        if x.dim() == 5:
            x = x.squeeze(-1).squeeze(-1)

        batch_size = x.shape[0]
        probs = torch.zeros(
            batch_size,
            self.num_channels,
            self.num_features,
            self.num_leaves,
            self.num_repetitions,
            device=x.device,
        )

        # Perform linear interpolation
        for i_feature in range(self.num_features):
            for i_channel in range(self.num_channels):
                for i_repetition in range(self.num_repetitions):
                    for i_leaf in range(self.num_leaves):
                        xs_i = self.xs[i_repetition][i_leaf][i_feature][i_channel]
                        ys_i = self.ys[i_repetition][i_leaf][i_feature][i_channel]
                        ivalues = interp(x[:, i_channel, i_feature], xs_i, ys_i)
                        probs[:, i_channel, i_feature, i_leaf, i_repetition] = ivalues

        # Return the logarithm of probabilities
        logprobs = torch.log(probs + 1e-10)
        logprobs = torch.clamp(logprobs, min=-300.0)
        return logprobs


class PiecewiseLinear(LeafModule):
    """Piecewise linear leaf distribution module.

    First constructs histograms from the data using K-means clustering,
    then approximates the histograms with piecewise linear functions.

    This leaf requires initialization with data via the `initialize()` method
    before it can be used for inference or sampling.

    Attributes:
        alpha: Laplace smoothing parameter.
        xs: Nested list of x-coordinates for piecewise linear functions.
        ys: Nested list of y-coordinates (densities) for piecewise linear functions.
        domains: List of Domain objects describing each feature.
        is_initialized: Whether the distribution has been initialized with data.
    """

    def __init__(
        self,
        scope: Scope | int | List[int],
        out_channels: int = 1,
        num_repetitions: int = 1,
        alpha: float = 0.0,
    ):
        """Initialize PiecewiseLinear leaf module.

        Args:
            scope: Variable scope (Scope, int, or list[int]).
            out_channels: Number of output channels (clusters via K-means).
            num_repetitions: Number of repetitions.
            alpha: Laplace smoothing parameter (default 0.0).
        """
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
        )

        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.alpha = alpha

        # These will be set during initialization
        self.xs: Optional[List] = None
        self.ys: Optional[List] = None
        self.domains: Optional[List[Domain]] = None
        self.is_initialized = False

        # Register a dummy parameter so device detection works
        self.register_buffer("_device_buffer", torch.zeros(1))

    @property
    def _torch_distribution_class(self):
        """PiecewiseLinear uses a custom distribution, not a torch.distributions class."""
        return None

    @property
    def _supported_value(self) -> float:
        """Returns a value in the support of the distribution."""
        return 0.0

    @property
    def distribution(self) -> PiecewiseLinearDist:
        """Returns the underlying PiecewiseLinearDist object.

        Raises:
            ValueError: If the distribution has not been initialized.
        """
        if not self.is_initialized:
            raise ValueError(
                "PiecewiseLinear leaf has not been initialized. "
                "Call initialize(data, domains) first."
            )
        return PiecewiseLinearDist(self.xs, self.ys, self.domains)  # type: ignore[arg-type]

    @property
    def mode(self) -> Tensor:
        """Return distribution mode.

        Returns:
            Mode of the distribution.
        """
        return self.distribution.mode

    def params(self) -> dict:
        """Returns the parameters of the distribution.

        For PiecewiseLinear, returns xs and ys nested lists.
        """
        return {"xs": self.xs, "ys": self.ys}

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> dict:
        """Not implemented for PiecewiseLinear - use initialize() instead."""
        raise NotImplementedError(
            "PiecewiseLinear does not support MLE. Use initialize() instead."
        )

    def initialize(self, data: Tensor, domains: List[Domain]) -> None:
        """Initialize the piecewise linear distribution with data.

        Uses K-means clustering to create multiple distributions per leaf,
        then constructs histograms and approximates them with piecewise
        linear functions.

        Args:
            data: Training data tensor of shape (N, F) where N is batch size
                and F is the number of features.
            domains: List of Domain objects, one per feature.

        Raises:
            ValueError: If data shape doesn't match scope.
        """
        try:
            from fast_pytorch_kmeans import KMeans
        except ImportError:
            raise ImportError(
                "fast_pytorch_kmeans required for PiecewiseLinear. "
                "Install with: pip install fast-pytorch-kmeans"
            )

        logger.info(f"Initializing PiecewiseLinear with data shape {data.shape}")

        # Validate input
        num_features = len(self.scope.query)
        if data.shape[1] != num_features:
            raise ValueError(
                f"Data has {data.shape[1]} features but scope has {num_features}"
            )

        if len(domains) != num_features:
            raise ValueError(
                f"Got {len(domains)} domains but scope has {num_features} features"
            )

        self.domains = domains
        device = data.device

        # Parameters stored as nested lists [R][L][F][C]
        xs = []
        ys = []
        num_leaves = self.out_shape.channels

        for i_repetition in range(self.out_shape.repetitions):
            xs_leaves = []
            ys_leaves = []

            # Cluster data into num_leaves clusters
            if num_leaves > 1:
                kmeans = KMeans(
                    n_clusters=num_leaves, mode="euclidean", verbose=0, init_method="random"
                )
                kmeans.fit(data.float())
                cluster_idxs = kmeans.max_sim(a=data.float(), b=kmeans.centroids)[1]
            else:
                cluster_idxs = torch.zeros(data.shape[0], dtype=torch.long, device=device)

            for cluster_idx in range(num_leaves):
                # Select data for this cluster
                mask = cluster_idxs == cluster_idx
                cluster_data = data[mask]

                xs_features = []
                ys_features = []

                for i_feature in range(num_features):
                    xs_channels = []
                    ys_channels = []

                    # For PiecewiseLinear, we use a single "channel" per feature
                    # (the reference used num_channels but SPFlow uses out_channels for leaves)
                    data_subset = cluster_data[:, i_feature].float()

                    if self.domains[i_feature].data_type == DataType.DISCRETE:
                        # Edges are the discrete values
                        mids = torch.tensor(
                            self.domains[i_feature].values, device=device
                        ).float()

                        # Add a break at the end
                        breaks = torch.cat(
                            [mids, torch.tensor([mids[-1] + 1], device=device)]
                        )

                        if data_subset.shape[0] == 0:
                            # If no data in cluster, use uniform
                            densities = torch.ones(len(mids), device=device) / len(mids)
                        else:
                            # Compute histogram densities
                            densities = torch.histogram(
                                data_subset.cpu(), bins=breaks.cpu(), density=True
                            ).hist.to(device)

                    elif self.domains[i_feature].data_type == DataType.CONTINUOUS:
                        # Find histogram bins using automatic bin width
                        if data_subset.numel() > 0:
                            bins, _ = get_bin_edges_torch(data_subset)
                        else:
                            # Fallback for empty data
                            bins = torch.linspace(
                                self.domains[i_feature].min or 0,
                                self.domains[i_feature].max or 1,
                                11,
                                device=device,
                            )

                        # Construct histogram
                        if data_subset.numel() > 0:
                            densities = torch.histogram(
                                data_subset.cpu(), bins=bins.cpu(), density=True
                            ).hist.to(device)
                        else:
                            densities = torch.ones(len(bins) - 1, device=device) / (
                                len(bins) - 1
                            )
                        breaks = bins
                        mids = ((breaks + torch.roll(breaks, shifts=-1, dims=0)) / 2)[:-1]
                    else:
                        raise ValueError(
                            f"Unknown data type: {domains[i_feature].data_type}"
                        )

                    # Apply optional Laplace smoothing
                    if self.alpha > 0:
                        n_samples = data_subset.shape[0]
                        n_bins = len(breaks) - 1
                        counts = densities * n_samples
                        densities = (counts + self.alpha) / (
                            n_samples + n_bins * self.alpha
                        )

                    # Add tail breaks to start and end
                    if self.domains[i_feature].data_type == DataType.DISCRETE:
                        tail_width = 1
                        x = [b.item() for b in breaks[:-1]]
                        x = [x[0] - tail_width] + x + [x[-1] + tail_width]
                    elif self.domains[i_feature].data_type == DataType.CONTINUOUS:
                        EPS = 1e-8
                        x = (
                            [breaks[0].item() - EPS]
                            + [b0.item() + (b1.item() - b0.item()) / 2 for (b0, b1) in pairwise(breaks)]
                            + [breaks[-1].item() + EPS]
                        )
                    else:
                        raise ValueError(
                            f"Unknown data type in tail break construction: {self.domains[i_feature].data_type}"
                        )

                    # Add density 0 at start and end tail breaks
                    y = [0.0] + [d.item() for d in densities] + [0.0]

                    # Construct tensors
                    x = torch.tensor(x, device=device, dtype=torch.float32)
                    y = torch.tensor(y, device=device, dtype=torch.float32)

                    # Compute AUC using the trapeziod rule
                    auc = torch.trapezoid(y=y, x=x)

                    # Normalize y to sum to 1 using AUC
                    if auc > 0:
                        y = y / auc

                    xs_channels.append(x)
                    ys_channels.append(y)

                    xs_features.append(xs_channels)
                    ys_features.append(ys_channels)

                xs_leaves.append(xs_features)
                ys_leaves.append(ys_features)

            xs.append(xs_leaves)
            ys.append(ys_leaves)

        self.xs = xs
        self.ys = ys
        self.is_initialized = True

        logger.info("PiecewiseLinear initialization complete")

    def reset(self) -> None:
        """Reset the distribution to uninitialized state."""
        self.is_initialized = False
        self.xs = None
        self.ys = None
        self.domains = None

    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log-likelihoods for input data.

        Args:
            data: Input data tensor of shape (N, F).
            cache: Optional cache dictionary.

        Returns:
            Log-likelihood tensor.
        """
        if not self.is_initialized:
            raise ValueError(
                "PiecewiseLinear leaf has not been initialized. "
                "Call initialize(data, domains) first."
            )

        if data.dim() != 2:
            raise ValueError(
                f"Data must be 2-dimensional (batch, num_features), got shape {data.shape}."
            )

        # Get scope-relevant data
        data_q = data[:, self.scope.query]

        # Handle marginalization
        marg_mask = torch.isnan(data_q)
        has_marginalizations = marg_mask.any()

        if has_marginalizations:
            data_q = data_q.clone()
            data_q[marg_mask] = self._supported_value

        # Unsqueeze to add channel dimension
        data_q = data_q.unsqueeze(1)  # [N, 1, F]

        # Compute log probabilities
        dist = self.distribution
        log_prob = dist.log_prob(data_q)

        # Marginalize entries
        if has_marginalizations:
            # Expand mask to match log_prob shape
            marg_mask_expanded = marg_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            marg_mask_expanded = torch.broadcast_to(marg_mask_expanded, log_prob.shape)
            log_prob[marg_mask_expanded] = 0.0

        return log_prob

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: Optional[SamplingContext] = None,
    ) -> Tensor:
        """Sample from the piecewise linear distribution.

        Args:
            num_samples: Number of samples to generate.
            data: Optional evidence tensor.
            is_mpe: Perform MPE (mode) instead of sampling.
            cache: Optional cache dictionary.
            sampling_ctx: Optional sampling context.

        Returns:
            Sampled data tensor.
        """
        if not self.is_initialized:
            raise ValueError(
                "PiecewiseLinear leaf has not been initialized. "
                "Call initialize(data, domains) first."
            )

        # Prepare data tensor
        data = self._prepare_sample_data(num_samples, data)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

        out_of_scope = list(
            filter(lambda x: x not in self.scope.query, range(data.shape[1]))
        )
        marg_mask = torch.isnan(data)
        marg_mask[:, out_of_scope] = False

        # Mask that tells us which feature at which sample is relevant
        samples_mask = marg_mask
        samples_mask[:, self.scope.query] &= sampling_ctx.mask

        # Count number of samples to draw
        instance_mask = samples_mask.sum(1) > 0
        n_samples = instance_mask.sum()

        if sampling_ctx.repetition_idx is None:
            if self.out_shape.repetitions > 1:
                raise ValueError(
                    "Repetition index must be provided in sampling context for leaves with multiple repetitions."
                )
            else:
                sampling_ctx.repetition_idx = torch.zeros(
                    data.shape[0], dtype=torch.long, device=data.device
                )

        dist = self.distribution
        n_samples_int = int(n_samples.item())

        if is_mpe:
            samples = dist.mode.unsqueeze(0)
            samples = samples.repeat(n_samples_int, 1, 1, 1, 1).detach()
        else:
            samples = dist.sample((n_samples_int,))

        # Handle repetition index
        if samples.ndim == 5:
            repetition_idx = sampling_ctx.repetition_idx[instance_mask]
            r_idxs = repetition_idx.view(-1, 1, 1, 1, 1).expand(
                -1, samples.shape[1], samples.shape[2], samples.shape[3], -1
            )
            samples = torch.gather(samples, dim=-1, index=r_idxs).squeeze(-1)

        # Handle channel index - gather on leaves dimension (dim=3)
        # samples shape after repetition handling: (N, C=1, F, L)
        if self.out_shape.channels == 1:
            sampling_ctx.channel_index.zero_()

        # c_idxs needs shape (N, 1, F, 1) to gather on dim=3
        c_idxs = sampling_ctx.channel_index[instance_mask]  # (N,)
        c_idxs = c_idxs.view(-1, 1, 1, 1).expand(-1, 1, samples.shape[2], 1)  # (N, 1, F, 1)
        samples = samples.gather(dim=3, index=c_idxs).squeeze(3)  # (N, 1, F)

        # Squeeze channel dimension
        samples = samples.squeeze(1)  # (N, F)

        # Update data with samples
        row_indices = instance_mask.nonzero(as_tuple=True)[0]
        scope_idx = torch.tensor(self.scope.query, dtype=torch.long, device=data.device)
        rows = row_indices.unsqueeze(1).expand(-1, len(scope_idx))
        cols = scope_idx.unsqueeze(0).expand(n_samples_int, -1)
        mask_subset = samples_mask[instance_mask][:, self.scope.query]

        data[rows[mask_subset], cols[mask_subset]] = samples[mask_subset].to(data.dtype)

        return data
