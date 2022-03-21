"""
Module that contains a set of distributions with learnable parameters.
"""
import logging
from abc import abstractmethod
from typing import List

import numpy as np
import torch as th
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from utils import SamplingContext
from clipper import DistributionClipper
from layers import AbstractLayer, Sum
from type_checks import check_valid

logger = logging.getLogger(__name__)


def dist_forward(distribution, x):
    """
    Forward pass with an arbitrary PyTorch distribution.

    Args:
        distribution: PyTorch base distribution which is used to compute the log probabilities of x.
        x: Input to compute the log probabilities of.
           Shape [n, d].

    Returns:
        th.Tensor: Log probabilities for each feature.
    """
    # Make room for out_channels and num_repetitions of layer
    if x.dim() == 3:  # Number of repetition dimension already exists
        x = x.unsqueeze(2)  # Shape [n, d, 1, r]
    elif x.dim() == 2:
        x = x.unsqueeze(2).unsqueeze(3)  # Shape: [n, d, 1, 1]

    # Compute log-likelihodd
    x = distribution.log_prob(x)  # Shape: [n, d, oc, r]

    return x


def _mode(distribution: dist.Distribution, context: SamplingContext = None) -> th.Tensor:
    """
    Get the mode of a given distribution.

    Args:
        distribution: Leaf distribution from which to choose the mode from.
        context: Sampling context.
    Returns:
        th.Tensor: Mode of the given distribution.
    """
    # TODO: Implement more torch distributions
    if isinstance(distribution, dist.Normal):
        # Repeat the mode along the batch axis
        return distribution.mean.repeat(context.n, 1, 1, 1, 1)
    elif isinstance(distribution, dist.Bernoulli):
        mode = distribution.probs.clone()
        mode[mode >= 0.5] = 1.0
        mode[mode < 0.5] = 0.0
        return mode.repeat(context.n, 1, 1, 1, 1)
    else:
        raise Exception(f"MPE not yet implemented for type {type(distribution)}")


def dist_sample(distribution: dist.Distribution, context: SamplingContext = None) -> th.Tensor:
    """
    Sample n samples from a given distribution.

    Args:
        repetition_indices: Indices into the repetition axis.
        distribution (dists.Distribution): Base distribution to sample from.
        parent_indices (th.Tensor): Tensor of indexes that point to specific representations of single features/scopes.
    """

    # Sample from the specified distribution
    if context.is_mpe:
        samples = _mode(distribution, context)
    else:
        nr_sampled_nodes, n = context.parent_indices.shape[:2]
        if distribution.has_rsample:
            samples = distribution.rsample(sample_shape=(nr_sampled_nodes, n))
        else:
            samples = distribution.sample(sample_shape=(nr_sampled_nodes, n))

    # w is the "weight sets" dimension. In the CSPN case, the weights are different for each conditional.
    # In the RatSpn case, w=1.
    nr_sampled_nodes, n, w, d, oc, r = samples.shape

    # Filter each sample by its specific repetition
    if context.repetition_indices is not None:
        rep_ind = context.repetition_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, d, oc, -1)
        samples = th.gather(samples, dim=-1, index=rep_ind).squeeze(-1)
    # tmp = th.zeros(n, d, c, device=context.repetition_indices.device)
    # for i in range(n):
        # tmp[i, :, :] = samples[i, :, :, context.repetition_indices[i]]
    # if context.repetition_indices is not None:
        # rep_selections = []
        # for i in range(n):
            # rep_selections.append(samples[i, range(w), :, :, context.repetition_indices[i]])
        # samples = th.stack(rep_selections)

    # If parent index into out_channels are given
    if context.parent_indices is not None:
        par_ind = context.parent_indices.unsqueeze(-1)
        samples = th.gather(samples, dim=-1, index=par_ind).squeeze(-1)
        # par_selected = []
        # for i in range(n):
            # Choose only specific samples for each feature/scope
            # par_selected.append(th.gather(samples[i], dim=2,
                                             # index=context.parent_indices[i].unsqueeze(-1)).squeeze(-1))
        # samples = th.stack(par_selected)

    return samples


class Leaf(AbstractLayer):
    """
    Abstract layer that maps each input feature into a specified
    representation, e.g. Gaussians.

    Implementing layers shall be valid distributions.

    If the input at a specific position is NaN, the variable will be marginalized.
    """

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """
        Create the leaf layer.

        Args:
            in_features: Number of input features.
            out_channels: Number of parallel representations for each input feature.
            num_repetitions: Number of parallel repetitions of this layer.
            dropout: Dropout probability.
        """
        super().__init__(in_features=in_features, num_repetitions=num_repetitions)
        self.in_features = check_valid(in_features, int, 1)
        self.out_channels = check_valid(out_channels, int, 1)
        self.num_repetitions = check_valid(num_repetitions, int, 1)
        dropout = check_valid(dropout, float, 0.0, 1.0)
        self.dropout = nn.Parameter(th.tensor(dropout), requires_grad=False)

        self.out_shape = f"(N, {in_features}, {out_channels})"

        # Marginalization constant
        self.marginalization_constant = nn.Parameter(th.zeros(1), requires_grad=False)

        # Dropout bernoulli
        self._bernoulli_dist = th.distributions.Bernoulli(probs=self.dropout)

    def _apply_dropout(self, x: th.Tensor) -> th.Tensor:
        # Apply dropout sampled from a bernoulli during training (model.train() has been called)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_indices] = 0.0
        return x

    def _marginalize_input(self, x: th.Tensor) -> th.Tensor:
        # Marginalize nans set by user
        x = th.where(~th.isnan(x), x, self.marginalization_constant)
        return x

    def forward(self, x):
        # Forward through base distribution
        d = self._get_base_distribution()
        x = dist_forward(d, x)
        x = self._marginalize_input(x)
        x = self._apply_dropout(x)

        return x

    @abstractmethod
    def _get_base_distribution(self) -> dist.Distribution:
        """Get the underlying torch distribution."""
        pass

    @abstractmethod
    def moments(self):
        """Get the mean, variance and third central moment (unnormalized skew)"""
        pass

    @abstractmethod
    def gradient(self, x: th.Tensor, order: int):
        """Get the gradient of order 1, ..., order at the point x"""
        pass

    def sample(self, context: SamplingContext = None) -> th.Tensor:
        """
        Perform sampling, given indices from the parent layer that indicate which of the multiple representations
        for each input shall be used.
        """
        d = self._get_base_distribution()
        samples = dist_sample(distribution=d, context=context)
        return samples

    def entropy(self) -> th.Tensor:
        d = self._get_base_distribution()
        return d.entropy()

    @property
    def _device(self):
        """Small hack to obtain the current device."""
        return self.dropout.device

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_channels={self.out_channels}, dropout={self.dropout}, out_shape={self.out_shape})"


class Normal(Leaf):
    """Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)

        # Create gaussian means and stds
        self.means = nn.Parameter(th.randn(1, in_features, out_channels, num_repetitions))
        self.stds = nn.Parameter(th.rand(1, in_features, out_channels, num_repetitions))
        self.gauss = dist.Normal(loc=self.means, scale=self.stds)

    def _get_base_distribution(self):
        return self.gauss


class Bernoulli(Leaf):
    """Bernoulli layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)

        # Create bernoulli parameters
        self.probs = nn.Parameter(th.randn(1, in_features, out_channels, num_repetitions))

    def _get_base_distribution(self):
        # Use sigmoid to ensure, that probs are in valid range
        probs_ratio = th.sigmoid(self.probs)
        return dist.Bernoulli(probs=probs_ratio)


class MultivariateNormal(Leaf):
    """Multivariate Gaussian layer."""

    def __init__(self, in_features: int, out_channels: int, cardinality: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.
            cardinality: Number of features covered.

        """
        # TODO: Fix for num_repetitions
        super().__init__(in_features, out_channels, num_repetitions, dropout)
        self.cardinality = check_valid(cardinality, int, 2, in_features + 1)
        self._pad_value = in_features % cardinality
        self._out_features = np.ceil(in_features / cardinality).astype(int)
        self._n_dists = np.ceil(in_features / cardinality).astype(int)

        # Create gaussian means and covs
        self.means = nn.Parameter(th.randn(out_channels * self._n_dists, cardinality))

        # Generate covariance matrix via the cholesky decomposition: s = A'A where A is a triangular matrix
        # Further ensure, that diag(a) > 0 everywhere, such that A has full rank
        rand = th.rand(out_channels * self._n_dists, cardinality, cardinality)

        # Make a matrices triangular
        for i in range(out_channels * self._n_dists):
            rand[i, :, :].tril_()

        self.triangular = nn.Parameter(rand)
        self._mv = dist.MultivariateNormal(loc=self.means, scale_tril=self.triangular)

        self.out_shape = f"(N, {self._out_features}, {self.out_channels})"

    def forward(self, x):
        # Pad dummy variable via reflection
        if self._pad_value != 0:
            x = F.pad(x, pad=[0, 0, 0, self._pad_value], mode="reflect")

        # Make room for out_channels of layer
        # Output shape: [n, 1, d]
        batch_size = x.shape[0]
        x = x.view(batch_size, self._n_dists, self.cardinality)
        x = x.repeat(1, self.out_channels, 1)

        # Compute multivariate gaussians
        # Output shape: [n, out_channels, d / cardinality]
        x = self._mv.log_prob(x)

        # Output shape: [n, d / cardinality, out_channels]
        x = x.view(batch_size, self._n_dists, self.out_channels)

        # Marginalize and apply dropout
        x = self._marginalize_input(x)
        x = self._apply_dropout(x)

        return x

    def sample(self, n: int = None, context: SamplingContext = None) -> th.Tensor:
        """TODO: Multivariate need special treatment."""
        raise Exception("Not yet implemented")

    def _get_base_distribution(self):
        return self._mv


class Beta(Leaf):
    """Beta layer. Maps each input feature to its beta log likelihood."""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a beta layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)

        # Create beta parameters
        self.concentration0 = nn.Parameter(th.rand(1, in_features, out_channels, num_repetitions))
        self.concentration1 = nn.Parameter(th.rand(1, in_features, out_channels, num_repetitions))
        self.beta = dist.Beta(concentration0=self.concentration0, concentration1=self.concentration1)

    def _get_base_distribution(self):
        return self.beta


class Cauchy(Leaf):
    """Cauchy layer. Maps each input feature to cauchy beta log likelihood."""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a cauchy layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)
        self.means = nn.Parameter(th.randn(1, in_features, out_channels, num_repetitions))
        self.stds = nn.Parameter(th.rand(1, in_features, out_channels, num_repetitions))
        self.cauchy = dist.Cauchy(loc=self.means, scale=self.stds)

    def _get_base_distribution(self):
        return self.cauchy


class Chi2(Leaf):
    """Chi square distribution layer"""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a chi square layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.
            num_repetitions: Number of parallel repetitions of this layer.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)
        self.df = nn.Parameter(th.rand(1, in_features, out_channels, num_repetitions))
        self.chi2 = dist.Chi2(df=self.df)

    def _get_base_distribution(self):
        return self.chi2


class Mixture(Leaf):
    def __init__(self, distributions, in_features: int, out_channels, num_repetitions, dropout=0.0):
        """
        Create a layer that stack multiple representations of a feature along the scope dimension.

        Args:
            distributions: List of possible distributions to represent the feature with.
            out_channels: out_channels of how many nodes each distribution is assigned to.
            in_features: Number of input features.
        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)
        # Build different layers for each distribution specified
        reprs = [distr(in_features, out_channels, num_repetitions, dropout) for distr in distributions]
        self.representations = nn.ModuleList(reprs)

        # Build sum layer as mixture of distributions
        self.sumlayer = Sum(
            in_features=in_features,
            in_channels=len(distributions) * out_channels,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
        )

    def _get_base_distribution(self):
        raise Exception("Not implemented")

    def forward(self, x):
        results = [d(x) for d in self.representations]

        # Stack along output channel dimension
        x = th.cat(results, dim=2)

        # Build mixture of different leafs per in_feature
        x = self.sumlayer(x)
        return x

    def sample(self, n: int = None, context: SamplingContext = None) -> th.Tensor:
        # Sample from sum mixture layer
        context = self.sumlayer.sample(context=context)

        # Collect samples from different distribution layers
        samples = []
        for d in self.representations:
            sample_d = d.sample(context=context)
            samples.append(sample_d)

        # Stack along channel dimension
        samples = th.cat(samples, dim=2)

        # If parent index into out_channels are given
        if context.parent_indices is not None:
            # Choose only specific samples for each feature/scope
            samples = th.gather(samples, dim=2, index=context.parent_indices.unsqueeze(-1)).squeeze(-1)

        return samples


class IsotropicMultivariateNormal(Leaf):
    """Isotropic multivariate gaussian layer.

    The covariance is simplified to:

    cov = sigma^2 * I

    Maps k input feature to their multivariate gaussian log likelihood."""

    def __init__(self, in_features, out_channels, num_repetitions, cardinality, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            cardinality: Number of features per gaussian.
            in_features: Number of input features.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)
        self.cardinality = cardinality

        # Number of different distributions: total number of features
        # divided by the number of features in each gaussian

        self._pad_value = in_features % cardinality
        self._out_features = np.ceil(in_features / cardinality).astype(int)

        self._n_dists = np.ceil(in_features / cardinality).astype(int)

        # Create gaussian means and stds
        self.means = nn.Parameter(th.randn(out_channels, self._n_dists, cardinality, num_repetitions))
        self.stds = nn.Parameter(th.rand(out_channels, self._n_dists, cardinality, num_repetitions))
        self.cov_factors = nn.Parameter(
            th.zeros(out_channels, self._n_dists, cardinality, num_repetitions), requires_grad=False
        )
        self.gauss = dist.LowRankMultivariateNormal(loc=self.means, cov_factor=self.cov_factors, cov_diag=self.stds)

    def forward(self, x):
        # TODO: Fix for num_repetitions

        # Pad dummy variable via reflection
        if self._pad_value != 0:
            # Do unsqueeze and squeeze due to padding not being allowed on 2D tensors
            x = x.unsqueeze(1)
            x = F.pad(x, pad=[0, self._pad_value // 2], mode="reflect")
            x = x.squeeze(1)

        # Make room for out_channels of layer
        # Output shape: [n, 1, d]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, self._n_dists, self.cardinality)

        # Compute multivariate gaussians
        # Output shape: [n, out_channels, d / cardinality]
        x = self.gauss.log_prob(x)

        # Output shape: [n, d / cardinality, out_channels]
        x = x.permute((0, 2, 1))

        x = self._marginalize_input(x)
        x = self._apply_dropout(x)

        return x

    def sample(self, n=None, context: SamplingContext = None) -> th.Tensor:
        """TODO: Multivariate need special treatment."""
        raise Exception("Not yet implemented")

    def _get_base_distribution(self):
        return self.gauss


class Gamma(Leaf):
    """Gamma distribution layer."""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a gamma layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)
        self.concentration = nn.Parameter(th.rand(1, in_features, out_channels, num_repetitions))
        self.rate = nn.Parameter(th.rand(1, in_features, out_channels, num_repetitions))
        self.gamma = dist.Gamma(concentration=self.concentration, rate=self.rate)

    def _get_base_distribution(self):
        return self.gamma


class Poisson(Leaf):
    """Poisson distribution layer."""

    def __init__(self, in_features: int, out_channels: int, num_repetitions: int = 1, dropout=0.0):
        """Creat a poisson layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)
        self.rate = nn.Parameter(th.rand(1, in_features, out_channels, num_repetitions))
        self.poisson = dist.Poisson(rate=self.rate)

    def _get_base_distribution(self):
        return self.poisson


if __name__ == "__main__":
    from layers import Product, Sum

    # Setup
    I = 3
    in_features = 2
    num_repetitions = 1
    batch_size = 1

    # Leaf layer: DistributionsMixture
    dists = [Gamma, Beta, Chi2, Cauchy]
    leaf = Mixture(distributions=dists, in_features=in_features, out_channels=I, num_repetitions=num_repetitions)
    # Add further layers
    pro1 = Product(in_features=in_features, cardinality=in_features, num_repetitions=num_repetitions)
    sum1 = Sum(in_features=1, in_channels=I, out_channels=1, num_repetitions=1)

    # Random input
    x = th.randn(batch_size, in_features)

    # Pass through leaf mixture layer
    x = leaf(x)

    # Check dimensions
    n, d, c, r = x.shape
    assert n == batch_size
    assert d == in_features
    assert c == I
    assert r == num_repetitions

    # Sample
    rep = th.zeros(5, dtype=int)
    idxs = sum1.sample(n=5)
    idxs = pro1.sample(indices=idxs)
    samples = leaf.sample(indices=idxs, repetition_indices=rep)
    assert (n, in_features) == samples.shape
    print(samples.shape)
