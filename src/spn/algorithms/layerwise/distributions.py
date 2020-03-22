"""
Module that contains a set of distributions with learnable parameters.
"""
import logging
from abc import abstractmethod

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from spn.algorithms.layerwise.clipper import DistributionClipper
from spn.algorithms.layerwise.layers import AbstractLayer
from spn.algorithms.layerwise.type_checks import check_valid

logger = logging.getLogger(__name__)


def dist_forward(distribution, x):
    """
    Forward pass with an arbitrary PyTorch distribution.

    Args:
        distribution: PyTorch base distribution which is used to compute the log probabilities of x.
        x: Input to compute the log probabilities of.
           Shape [n, d].

    Returns:
        torch.Tensor: Log probabilities for each feature.
    """
    # Make room for multiplicity of layer
    # Output shape: [n, d, 1]
    x = x.unsqueeze(2)

    # Compute gaussians
    # Output shape: [n, d, multiplicity]
    x = distribution.log_prob(x)

    return x


def dist_sample(distribution: dist.Distribution, parent_indices: torch.Tensor) -> torch.Tensor:
    """
    Sample n samples from a given distribution.

    Args:
        distribution (dists.Distribution): Base distribution to sample from.
        parent_indices (torch.Tensor): Tensor of indexes that point to specific representations of single features/scopes.
    """

    # Sample from the specified distribution
    samples = distribution.sample(sample_shape=(parent_indices.shape[0],))

    assert (
        samples.shape[1] == 1
    ), "Something went wrong. First sample size dimension should be size 1 due to the distribution parameter dimensions. Please report this issue."
    samples.squeeze_(1)

    # If parent index into multiplicity are given
    if parent_indices is not None:
        # Choose only specific samples for each feature/scope
        samples = torch.gather(samples, dim=2, index=parent_indices.unsqueeze(-1)).squeeze(-1)

    return samples


class Leaf(AbstractLayer):
    """
    Abstract layer that maps each input feature into a specified
    representation, e.g. Gaussians.

    Implementing layers shall be valid distributions.

    If the input at a specific position is NaN, the variable will be marginalized.
    """

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """
        Create the leaf layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.
            droptout: Dropout probabilities.
        """
        super().__init__()
        self.multiplicity = check_valid(multiplicity, int, 1)
        self.in_features = check_valid(in_features, int, 1)
        dropout = check_valid(dropout, float, 0.0, 1.0)
        self.dropout = nn.Parameter(torch.tensor(dropout), requires_grad=False)

        self.out_shape = f"(N, {in_features}, {multiplicity})"

        # Marginalization constant
        self.marginalization_constant = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Dropout bernoulli
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=self.dropout)

    def _apply_dropout(self, x: torch.Tensor) -> torch.Tensor:
        # Apply dropout sampled from a bernoulli during training (model.train() has been called)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_indices] = 0.0
        return x

    def _marginalize_input(self, x: torch.Tensor) -> torch.Tensor:
        # Marginalize nans set by user
        x = torch.where(~torch.isnan(x), x, self.marginalization_constant)
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

    def sample(self, n: int = 1, indices: torch.Tensor = None) -> torch.Tensor:
        """
        Perform sampling, given indices from the parent layer that indicate which of the multiple representations
        for each input shall be used.
        """
        d = self._get_base_distribution()
        samples = dist_sample(d, indices)
        return samples

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, multiplicity={self.multiplicity}, dropout={self.dropout}, out_shape={self.out_shape})"


class Normal(Leaf):
    """Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(1, in_features, multiplicity))
        self.stds = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.gauss = dist.Normal(loc=self.means, scale=self.stds)

    def _get_base_distribution(self):
        return self.gauss


class Bernoulli(Leaf):
    """Bernoulli layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)

        # Create bernoulli parameters
        self.probs = nn.Parameter(torch.randn(1, in_features, multiplicity))

    def _get_base_distribution(self):
        # Use sigmoid to ensure, that probs are in valid range
        probs_ratio = torch.sigmoid(self.probs)
        return dist.Bernoulli(probs=probs_ratio)


class MultivariateNormal(Leaf):
    """Multivariate Gaussian layer."""

    def __init__(self, multiplicity, in_features, cardinality, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.
            cardinality: Number of features covered.

        """
        super().__init__(multiplicity, in_features, dropout)
        self.cardinality = check_valid(cardinality, int, 2, in_features + 1)
        self._pad_value = in_features % cardinality
        self._out_features = np.ceil(in_features / cardinality).astype(int)
        self._n_dists = np.ceil(in_features / cardinality).astype(int)

        # Create gaussian means and covs
        self.means = nn.Parameter(torch.randn(multiplicity * self._n_dists, cardinality))

        # Generate covariance matrix via the cholesky decomposition: s = A'A where A is a triangular matrix
        # Further ensure, that diag(a) > 0 everywhere, such that A has full rank
        rand = torch.rand(multiplicity * self._n_dists, cardinality, cardinality)

        # Make a matrices triangular
        for i in range(multiplicity * self._n_dists):
            rand[i, :, :].tril_()

        self.triangular = nn.Parameter(rand)
        self._mv = dist.MultivariateNormal(loc=self.means, scale_tril=self.triangular)

        self.out_shape = f"(N, {self._out_features}, {self.multiplicity})"

    def forward(self, x):
        # Pad dummy variable via reflection
        if self._pad_value != 0:
            x = F.pad(x, pad=[0, 0, 0, self._pad_value], mode="reflect")

        # Make room for multiplicity of layer
        # Output shape: [n, 1, d]
        batch_size = x.shape[0]
        x = x.view(batch_size, self._n_dists, self.cardinality)
        x = x.repeat(1, self.multiplicity, 1)

        # Compute multivariate gaussians
        # Output shape: [n, multiplicity, d / cardinality]
        x = self._mv.log_prob(x)

        # Output shape: [n, d / cardinality, multiplicity]
        x = x.view(batch_size, self._n_dists, self.multiplicity)

        # Marginalize and apply dropout
        x = self._marginalize_input(x)
        x = self._apply_dropout(x)

        return x

    def sample(self, n: int = 1, indices=None) -> torch.Tensor:
        """TODO: Multivariate need special treatment."""
        raise Exception("Not yet implemented")

    def _get_base_distribution(self):
        return self._mv


class Beta(Leaf):
    """Beta layer. Maps each input feature to its beta log likelihood."""

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """Creat a beta layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)

        # Create beta parameters
        self.concentration0 = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.concentration1 = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.beta = dist.Beta(concentration0=self.concentration0, concentration1=self.concentration1)

    def _get_base_distribution(self):
        return self.beta


class Cauchy(Leaf):
    """Cauchy layer. Maps each input feature to cauchy beta log likelihood."""

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """Creat a cauchy layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_channels: Number of input channels.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)
        self.means = nn.Parameter(torch.randn(1, in_features, multiplicity))
        self.stds = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.cauchy = dist.Cauchy(loc=self.means, scale=self.stds)

    def _get_base_distribution(self):
        return self.cauchy


class Chi2(Leaf):
    """Chi square distribution layer"""

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """Creat a chi square layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)
        self.df = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.chi2 = dist.Chi2(df=self.df)

    def _get_base_distribution(self):
        return self.chi2


class Representations(Leaf):
    def __init__(self, distributions, multiplicity, in_features, dropout=0.0):
        """
        Create a layer that stack multiple representations of a feature along the scope dimension.

        Args:
            distributions: List of possible distributions to represent the feature with.
            multiplicity: Multiplicity of how many nodes each distribution is assigned to.
            in_features: Number of input features.
        """
        super().__init__(multiplicity, in_features, dropout)
        reprs = [distr(multiplicity, in_features, dropout) for distr in distributions]
        self.representations = nn.ModuleList(reprs)

    def forward(self, x):
        results = [d(x) for d in self.representations]

        # Stack along output channel dimension
        x = torch.cat(results, dim=1)
        return x

    def sample(self, n:int=1, indices:torch.Tensor=None) -> torch.Tensor:
        """TODO: Needs special treatment"""
        raise Exception("Not yet implemented")


class IsotropicMultivariateNormal(Leaf):
    """Isotropic multivariate gaussian layer.

    The covariance is simplified to:

    cov = sigma^2 * I

    Maps k input feature to their multivariate gaussian log likelihood."""

    def __init__(self, multiplicity, cardinality, in_features, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            cardinality: Number of features per gaussian.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)
        self.cardinality = cardinality

        # Number of different distributions: total number of features
        # divided by the number of features in each gaussian

        self._pad_value = in_features % cardinality
        self._out_features = np.ceil(in_features / cardinality).astype(int)

        self._n_dists = np.ceil(in_features / cardinality).astype(int)

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(multiplicity, self._n_dists, cardinality))
        self.stds = nn.Parameter(torch.rand(multiplicity, self._n_dists, cardinality))
        self.cov_factors = nn.Parameter(torch.zeros(multiplicity, self._n_dists, cardinality, 1), requires_grad=False)
        self.gauss = dist.LowRankMultivariateNormal(loc=self.means, cov_factor=self.cov_factors, cov_diag=self.stds)

    def forward(self, x):

        # Pad dummy variable via reflection
        if self._pad_value != 0:
            # Do unsqueeze and squeeze due to padding not being allowed on 2D tensors
            x = x.unsqueeze(1)
            x = F.pad(x, pad=[0, self._pad_value // 2], mode="reflect")
            x = x.squeeze(1)

        # Make room for multiplicity of layer
        # Output shape: [n, 1, d]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, self._n_dists, self.cardinality)

        # Compute multivariate gaussians
        # Output shape: [n, multiplicity, d / cardinality]
        x = self.gauss.log_prob(x)

        # Output shape: [n, d / cardinality, multiplicity]
        x = x.permute((0, 2, 1))

        x = self._marginalize_input(x)
        x = self._apply_dropout(x)

        return x

    def sample(self, n=None, indices=None) -> torch.Tensor:
        """TODO: Multivariate need special treatment."""
        raise Exception("Not yet implemented")

    def _get_base_distribution(self):
        return self.gauss


class Gamma(Leaf):
    """Gamma distribution layer."""

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """Creat a gamma layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)
        self.concentration = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.rate = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.gamma = dist.Gamma(concentration=self.concentration, rate=self.rate)

    def _get_base_distribution(self):
        return self.gamma


class Poisson(Leaf):
    """Poisson distribution layer."""

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """Creat a poisson layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)
        self.rate = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.poisson = dist.Poisson(rate=self.rate)

    def _get_base_distribution(self):
        return self.poisson


if __name__ == "__main__":
    # Test marginalization
    mask = torch.randn(784) < -100.0
    mask[[0, 1, 2]] = 1
    layer = Normal(1, 784)
    x = torch.randn(1, 784)
    x[:, mask] = float("nan")
    res = layer(x)
    assert res[0, 0, 0] == 0, "was " + str(res[0, 0, 0])
    assert res[0, 1, 0] == 0, "was " + str(res[0, 1, 0])
    assert res[0, 2, 0] == 0, "was " + str(res[0, 2, 0])
    assert res[0, 3, 0] != 0, "was " + str(res[0, 3, 0])
    exit()

    # Define the problem size
    batch_size = 10
    n_features = 3

    # How many different representations of that distribution
    multiplicity = 1

    # Target probs to be learned
    probs = torch.tensor([0.1, 0.3, 0.5])
    bern = dist.Bernoulli(probs)

    # Bernoulli layer
    layer = Bernoulli(multiplicity, n_features)

    # Clipper to keep probs in [0, 1]
    clipper = DistributionClipper()

    # Use SGD
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
    for i in range(10000):
        x = bern.sample([batch_size])
        # Reset gradients
        optimizer.zero_grad()

        # Inference
        output = layer(x)

        # Comput loss
        loss = -1 * output.mean()

        # Backprop
        loss.backward(retain_graph=True)
        optimizer.step()
        layer.apply(clipper)

        print(loss)
        print(layer.probs)

    exit()

    # Create MV distribution to sample from
    loc1 = torch.rand(n_features // 2)
    loc2 = torch.rand(n_features // 2)
    locs = [loc1, loc2]
    triang1 = torch.tril(torch.rand(n_features // 2, n_features // 2))
    triang2 = torch.tril(torch.rand(n_features // 2, n_features // 2))
    triangs = [triang1, triang2]
    mv1 = dist.MultivariateNormal(loc=loc1, scale_tril=triang1)
    mv2 = dist.MultivariateNormal(loc=loc2, scale_tril=triang2)

    # Multivariate normal layer for SPNs
    model = MultivariateNormal(multiplicity=multiplicity, in_features=n_features, cardinality=n_features // 2)

    # Use SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def print_true_error(i):
        print(
            f"Squared error on means [{i}] :",
            torch.pow(model.means.view(multiplicity, model._n_dists, -1)[:, i, :] - locs[i], 2).sum().item(),
        )
        print(
            f"Squared error on tril [{i}] : ",
            torch.pow(
                model.triangular.view(multiplicity, model._n_dists, 3, 3)[:, i, :, :] - triangs[i].view(1, 3, 3), 2
            )
            .sum()
            .item(),
        )

    # Train for 100 epochs
    for epoch in range(100):

        loss_fn = nn.NLLLoss()
        for batch_index in range(1000):
            # Send data to correct device
            data1, data2 = mv1.sample([batch_size]), mv2.sample([batch_size])

            data = torch.stack([data1, data2], 1)

            # Reset gradients
            optimizer.zero_grad()

            # Inference
            output = model(data)

            # Comput loss
            loss = -1 * output.mean()

            # Backprop
            loss.backward()
            optimizer.step()

            # Log stuff
            if batch_index % 100 == 0:
                print(
                    "Train Epoch: {} [{: >5}/{: <5} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch, batch_index * len(data), 100 * 1000, 100.0 * batch_index / 1000, loss.item() / 100
                    )
                )
                print_true_error(0)
                print_true_error(1)
