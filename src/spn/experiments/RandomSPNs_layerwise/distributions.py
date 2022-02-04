import logging
from typing import Dict, Tuple

import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from spn.algorithms.layerwise.distributions import Leaf, dist_forward
from spn.algorithms.layerwise.layers import Product, Sum
from spn.algorithms.layerwise.type_checks import check_valid
from spn.algorithms.layerwise.utils import SamplingContext

logger = logging.getLogger(__name__)


class RatNormal(Leaf):
    """ Implementation as in RAT-SPN

    Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(
        self,
        in_features: int,
        out_channels: int,
        num_repetitions: int = 1,
        dropout: float = 0.0,
        min_sigma: float = 0.1,
        max_sigma: float = 1.0,
        min_mean: float = None,
        max_mean: float = None,
    ):
        """Creat a gaussian layer.

        Args:
            out_channels: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(1, in_features, out_channels, num_repetitions))

        if min_sigma is not None and max_sigma is not None:
            # Init from normal
            self.stds = nn.Parameter(torch.randn(1, in_features, out_channels, num_repetitions))
        else:
            # Init uniform between 0 and 1
            self.stds = nn.Parameter(torch.rand(1, in_features, out_channels, num_repetitions))

        self.min_sigma = check_valid(min_sigma, float, 0.0, max_sigma)
        self.max_sigma = check_valid(max_sigma, float, min_sigma)
        self.min_mean = check_valid(min_mean, float, upper_bound=max_mean, allow_none=True)
        self.max_mean = check_valid(max_mean, float, min_mean, allow_none=True)

    def bounded_dist_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.min_sigma < self.max_sigma:
            sigma_ratio = torch.sigmoid(self.stds)
            sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma_ratio
        else:
            sigma = 1.0

        means = self.means
        if self.max_mean:
            assert self.min_mean is not None
            mean_range = self.max_mean - self.min_mean
            means = torch.sigmoid(self.means) * mean_range + self.min_mean

        return means, sigma

    def _get_base_distribution(self) -> torch.distributions.Distribution:
        means, sigma = self.bounded_dist_params()
        gauss = dist.Normal(means, torch.sqrt(sigma))
        return gauss

    def moments(self):
        """Get the mean, variance and third central moment (unnorm. skew), which is 0"""
        means, sigma = self.bounded_dist_params()
        return means, sigma

    def gradient(self, x: torch.Tensor, order: int):
        """Get the gradient up to a given order at the point x"""
        assert order <= 3, "Gradient only implemented up to the third order!"

        x.unsqueeze_(2)  # shape [n, feat, 1, r]
        d = self._get_base_distribution()
        prob = d.log_prob(x).exp_()
        inv_var = 1/d.variance
        a = inv_var * (x - d.mean)
        grads = []
        if order >= 1:
            grads += [- prob * a]
        if order >= 2:
            grads += [prob * (a**2 - inv_var)]
        if order >= 3:
            grads += [prob * (-a**3 + 3 * a * inv_var)]
        return grads


class IndependentMultivariate(Leaf):
    def __init__(
        self,
        in_features: int,
        out_channels: int,
        cardinality: int,
        num_repetitions: int = 1,
        dropout: float = 0.0,
        leaf_base_class: Leaf = RatNormal,
        leaf_base_kwargs: Dict = None,
    ):
        """
        Create multivariate distribution that only has non zero values in the covariance matrix on the diagonal.

        Args:
            out_channels: Number of parallel representations for each input feature.
            cardinality: Number of variables per gauss.
            in_features: Number of input features.
            dropout: Dropout probabilities.
            leaf_base_class (Leaf): The encapsulating base leaf layer class.

        """
        super(IndependentMultivariate, self).__init__(in_features, out_channels, num_repetitions, dropout)
        if leaf_base_kwargs is None:
            leaf_base_kwargs = {}

        self.base_leaf = leaf_base_class(
            out_channels=out_channels,
            in_features=in_features,
            dropout=dropout,
            num_repetitions=num_repetitions,
            **leaf_base_kwargs,
        )
        self._pad = (cardinality - self.in_features % cardinality) % cardinality
        # Number of input features for the product needs to be extended depending on the padding applied here
        prod_in_features = in_features + self._pad
        self.prod = Product(
            in_features=prod_in_features, cardinality=cardinality, num_repetitions=num_repetitions
        )

        self.cardinality = check_valid(cardinality, int, 1, in_features + 1)
        self.out_shape = f"(N, {self.prod._out_features}, {out_channels}, {self.num_repetitions})"

    @property
    def out_features(self):
        return self.prod._out_features

    def _init_weights(self):
        if isinstance(self.base_leaf, RatNormal):
            truncated_normal_(self.base_leaf.stds, std=0.5)

    def pad_input(self, x: torch.Tensor):
        if self._pad:
            x = F.pad(x, pad=[0, 0, 0, 0, 0, self._pad], mode="constant", value=0.0)
        return x

    @property
    def pad(self):
        return self._pad

    def forward(self, x: torch.Tensor, reduction='sum'):
        # Pass through base leaf
        x = self.base_leaf(x)
        x = self.pad_input(x)

        # Pass through product layer
        x = self.prod(x, reduction=reduction)
        return x

    def _get_base_distribution(self):
        raise Exception("IndependentMultivariate does not have an explicit PyTorch base distribution.")

    def sample(self, n: int = None, context: SamplingContext = None) -> torch.Tensor:
        context = self.prod.sample(context=context)

        # Remove padding
        if self._pad:
            context.parent_indices = context.parent_indices[:, : -self._pad]

        samples = self.base_leaf.sample(context=context)
        return samples

    def moments(self):
        """Get the mean, variance and third central moment (unnormalized skew)"""
        return [self.prod(self.pad_input(m), reduction=None) for m in self.base_leaf.moments()]

    def gradient(self, x: torch.Tensor, order: int):
        """Get the gradient up to the given order at the point x"""
        return [self.prod(self.pad_input(g), reduction=None) for g in self.base_leaf.gradient(x, order)]

    def __repr__(self):
        return f"IndependentMultivariate(in_features={self.in_features}, out_channels={self.out_channels}, dropout={self.dropout}, cardinality={self.cardinality}, out_shape={self.out_shape})"


class GaussianMixture(IndependentMultivariate):
    def __init__(
            self,
            in_features: int,
            gmm_modes: int,
            out_channels: int,
            cardinality: int,
            num_repetitions: int = 1,
            dropout: float = 0.0,
            leaf_base_class: Leaf = RatNormal,
            leaf_base_kwargs: Dict = None,
    ):
        super(GaussianMixture, self).__init__(
            in_features=in_features, out_channels=gmm_modes, cardinality=cardinality, num_repetitions=num_repetitions,
            dropout=dropout, leaf_base_class=leaf_base_class, leaf_base_kwargs=leaf_base_kwargs)
        self.sum = Sum(in_features=self.out_features, in_channels=gmm_modes, num_repetitions=num_repetitions,
                       out_channels=out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, reduction='sum'):
        x = super().forward(x=x, reduction=reduction)
        x = self.sum(x)
        return x

    def sample(self, n: int = None, context: SamplingContext = None) -> torch.Tensor:
        context = self.sum.sample(context=context)
        return super(GaussianMixture, self).sample(context=context)

    def moments(self):
        """Get the mean, variance and third central moment (unnormalized skew)"""
        dist_moments = [self.prod(self.pad_input(m), reduction=None) for m in self.base_leaf.moments()]
        weights = self.sum.weights.unsqueeze(2)
        # Weights is of shape [n, d, 1, ic, oc, r]
        # Create an extra dimension for the mean vector so all elements of the mean vector are multiplied by the same
        # weight for that feature and output channel.

        child_mean = dist_moments[0]
        # moments have shape [n, d, cardinality, ic, r]
        # Create an extra 'output channels' dimension, as the weights are separate for each output channel.
        child_mean.unsqueeze_(4)
        mean = child_mean * weights
        # mean has shape [n, d, cardinality, ic, oc, r]
        mean = mean.sum(dim=3)
        # mean has shape [n, d, cardinality, oc, r]
        moments = [mean]

        centered_mean = child_var = 0
        if len(dist_moments) >= 2:
            child_var = dist_moments[1]
            child_var.unsqueeze_(4)
            centered_mean = child_mean - mean.unsqueeze(4)
            var = child_var + centered_mean**2
            var = var * weights
            var = var.sum(dim=3)
            moments += [var]

        if len(dist_moments) >= 3:
            child_skew = dist_moments[2]
            skew = 3 * centered_mean * child_var + centered_mean ** 3
            if child_skew is not None:
                child_skew.unsqueeze_(4)
                skew = skew + child_skew
            skew = skew * weights
            skew = skew.sum(dim=3)
            moments += [skew]

        # layer.mean, layer.var, layer.skew = mean, var, skew
        return moments

    def gradient(self, x: torch.Tensor, order: int):
        """Get the gradient up to the given order at the point x"""
        weights = self.sum.weights.unsqueeze(2)
        # Weights is of shape [n, d, 1, ic, oc, r]
        # The extra dimension is created so all elements of the gradient vectors are multiplied by the same
        # weight for that feature and output channel.
        grads = [self.prod(self.pad_input(g), reduction=None).unsqueeze(4) * weights
                 for g in self.base_leaf.gradient(x, order)]
        return grads

    def __repr__(self):
        return f"IndependentGMM(in_features={self.in_features}, out_channels={self.out_channels}, dropout={self.dropout}, cardinality={self.cardinality}, out_shape={self.out_shape})"


def truncated_normal_(tensor, mean=0, std=0.1):
    """
    Truncated normal from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
