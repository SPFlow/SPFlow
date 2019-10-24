import logging

import numpy as np
import tensorflow as tf
import torch
from torch import nn
from torch.nn import functional as F
import time

from torch import distributions as dist
from spn.algorithms.layerwise.distributions import Normal, dist_forward
from spn.algorithms.layerwise.distributions import IsotropicMultivariateNormal
from spn.algorithms.layerwise.distributions import MultivariateNormal
from spn.algorithms.layerwise.distributions import Leaf
from spn.algorithms.layerwise.layers import Product, Sum

logger = logging.getLogger(__name__)


class RegionSpn(nn.Module):
    """Defines a single SPN that is create via RatSpnConstructor.random_split(...)."""

    def __init__(self, S, I, dropout, num_parts, num_recursions, in_features):
        """
        Init a Region SPN split.

        Args:
            S (int): Number of sum nodes.
            I (int): Number of distributions for each leaf node.
            dropout (float): Dropout probability.
            num_parts (int): Number of partitions.
            num_recursions (int): Number of split repetitions.

        """
        super().__init__()

        self.S = S
        self.I = I
        self.dropout = dropout
        self.num_parts = num_parts
        self.num_recursions = num_recursions
        self.in_features = in_features
        self._leaf_output_features = num_parts ** num_recursions

        # Build RegionSpn
        self._build()

        # Randomize features
        self.rand_idxs = torch.tensor(np.random.permutation(in_features))

    def _build(self):
        # Build the SPN bottom up:

        # Definition from RAT Paper
        # Leaf Region:      Create I leaf nodes
        # Root Region:      Create C sum nodes
        # Internal Region:  Create S sum nodes
        # Partition:        Cross products of all child-regions

        ### LEAF ###
        # Cardinality is the size of the region in the last partitions
        cardinality = np.ceil(self.in_features / (self.num_parts ** self.num_recursions)).astype(int)
        self._leaf = IndependentNormal(
            multiplicity=self.I, in_features=self.in_features, cardinality=cardinality, dropout=self.dropout
        )
        self._inner_layers = nn.Sequential()

        count = 0
        prod = RatProduct(in_features=self.num_parts ** self.num_recursions)
        self._inner_layers.add_module(f"Product-{count}", prod)
        count += 1

        for i in np.arange(start=self.num_recursions - 1, stop=0, step=-1):
            is_lowest_sum_layer = i == self.num_recursions - 1
            if is_lowest_sum_layer:
                # Output channels channels of product layer after leafs
                sum_in_channels = self.I ** 2
            else:
                # Output channels of product layer after sums
                sum_in_channels = self.S ** 2

            in_features = self.num_parts ** i

            # Sum layer
            sumlayer = Sum(in_features=in_features, in_channels=sum_in_channels, out_channels=self.S)

            # Product layer
            prod = RatProduct(in_features=in_features)

            # Collect
            self._inner_layers.add_module(f"Sum-{count}", sumlayer)
            self._inner_layers.add_module(f"Product-{count}", prod)
            count += 1

    def forward(self, x):
        # Random permutation
        x = x[:, self.rand_idxs]

        # Apply leaf distributions
        x = self._leaf(x)

        # Forward to inner product and sum layers
        x = self._inner_layers(x)
        return x


class RatSpnConstructor:
    def __init__(self, in_features, C, S, I, dropout=0.0):
        """
        RAT SPN class.

        Parameters according to the paper (see Args for descriptions).
        
        Args:
            in_features (int): Number of input features.
            C (int): Number of classes.
            S (int): Number of sum nodes.
            I (int): Number of distributions for each leaf node.
            dropout (float): Dropout probability.
        """
        self.in_features = in_features
        self.C = C
        self.S = S
        self.I = I
        self.dropout = dropout

        # Collect SPNs. Each random_split(...) call adds one SPN
        self._region_spns = []

    def _create_spn(self, num_parts, num_recursions=1):
        """Create an SPN from the given RAT parameters.
        
        Args:
            num_parts (int): Number of partitions.
            num_recursions (int, optional): Number of split repetitions. Defaults to 1.
        """

        spn = RegionSpn(self.S, self.I, self.dropout, num_parts, num_recursions, self.in_features)

        self._region_spns.append(spn)

    def random_split(self, num_parts, num_recursions=1):
        """Randomly split the region graph.
        
        Args:
            num_parts (int): Number of partitions.
            num_recursions (int, optional): Number of split repetitions. Defaults to 1.
        """

        if num_parts ** (num_recursions) > self.in_features:
            raise Exception(
                f"The values for num_parts ({num_parts}) and num_recursions ({num_recursions}) have to satisfiy the condition 'num_parts ** num_recursions ({num_parts ** num_recursions}) <= in_features ({self.in_features})'"
            )
        self._create_spn(num_parts, num_recursions)

    def build(self):
        """Build the RatSpn object from the defined region graph"""
        if len(self._region_spns) == 0:
            raise Exception(
                "No random split has been added. Call random_split(...) at least once before building the RatSpn."
            )
        return RatSpn(region_spns=self._region_spns, C=self.C, S=self.S, I=self.I)


class RatNormal(Leaf):
    """ Implementation as in RAT-SPN

    Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(
        self, multiplicity, in_features, dropout=0.0, min_sigma=0.1, max_sigma=1.0, min_mean=None, max_mean=None
    ):
        """Creat a gaussian layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(1, in_features, multiplicity))
        self.stds = nn.Parameter(torch.rand(1, in_features, multiplicity))

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_mean = min_mean
        self.max_mean = max_mean

    def forward(self, x):
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

        gauss = dist.Normal(means, torch.sqrt(sigma))
        x = dist_forward(gauss, x)
        x = super().forward(x)
        return x


class IndependentNormal(Leaf):
    def __init__(self, multiplicity, in_features, cardinality, dropout=0.0):
        """
        Create multivariate normal that only has non zero values in the covariance matrix on the diagonal.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            cardinality: Number of variables per gauss.
            in_features: Number of input features.
            droptout: Dropout probabilities.
        """
        super(IndependentNormal, self).__init__(multiplicity, in_features, dropout)
        self.gauss = RatNormal(multiplicity=multiplicity, in_features=in_features, dropout=dropout)
        self.prod = Product(in_features=in_features, cardinality=cardinality)
        self._pad = (cardinality - self.in_features % cardinality) % cardinality

        self.cardinality = cardinality
        self.out_shape = f"(N, {self.prod._out_features}, {multiplicity})"

    def _init_weights(self):
        """Iniialize Normal weights with a truncated distribution."""
        truncated_normal_(self.gauss.stds, std=0.5)

    def forward(self, x):
        x = self.gauss(x)
        x = torch.where(~torch.isnan(x), x, torch.zeros(1).to(x.device))

        if self._pad:
            # Pad marginalized node
            x = F.pad(x, pad=[0, 0, 0, self._pad], mode="constant", value=0.0)

        x = self.prod(x)
        return x

    def __repr__(self):
        return f"IndependentNormal(in_features={self.in_features}, multiplicity={self.multiplicity}, dropout={self.dropout}, cardinality={self.cardinality}, out_shape={self.out_shape})"


class RatProduct(nn.Module):
    """
    Layerwise implementation of a RAT Product node.

    Builds the the combination of all children in two regions:
    res = []
    for n1 in R1, n2 in R2:
        res += [n1 * n2]
        
    TODO: Generalize to k regions.
    """

    def __init__(self, in_features):
        """
        Create a rat product node layer.

        Args:
            in_features (int): Number of input features.
            randomize (bool): Whether to randomize the selection of scopes.
                If false, scopes are chosen consecutively.
        """

        super().__init__()
        cardinality = 2
        self.in_features = in_features
        self.cardinality = cardinality
        in_features = int(in_features)
        self._cardinality = cardinality
        self._out_features = np.ceil(in_features / cardinality).astype(int)

        # Collect scopes for each product child
        self._scopes = [[] for _ in range(cardinality)]
        # Create random sequence of scopes
        scopes = np.random.permutation(in_features)

        # For two consecutive (random) scopes
        for i in range(0, in_features, cardinality):
            for j in range(cardinality):
                if i + j < in_features:
                    self._scopes[j].append(scopes[i + j])
                else:
                    # Case: d mod cardinality != 0 => Create marginalized nodes with prob 1.0
                    # Pad x in forward pass on the right: [n, d, c] -> [n, d+1, c] where index
                    # d+1 is the marginalized node (index "in_features")
                    self._scopes[j].append(in_features)

        # Transform into numpy array for easier indexing
        self._scopes = np.array(self._scopes)

        self.out_shape = f"(N, {self._out_features}, C_in^2)"

    def forward(self, x):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [batch, in_features, channel].

        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/2), channel * channel].
        """
        # Check if padding to next power of 2 is necessary
        if self.in_features != x.shape[1]:
            # Compute necessary padding to the next power of 2
            pad = 2 ** np.ceil(np.log2(x.shape[1])).astype(np.int) - x.shape[1]

            # Pad marginalized node
            x = F.pad(x, pad=[0, 0, 0, pad], mode="constant", value=0.0)

        # Create zero tensor and sum up afterwards
        batch = x.shape[0]
        channels = x.shape[2]
        result = torch.zeros(batch, self._out_features, channels).to(x.device)

        # Build outer sum, using broadcasting, this can be done with
        # modifying the tensor timensions:
        # left: [n, d, c] -> [n, d, c, 1]
        # right: [n, d, c] -> [n, d, 1, c]
        # left + right with broadcasting: [n, d, c, 1] + [n, d, 1, c] -> [n, d, c, c]
        left = x[:, self._scopes[0, :], :].unsqueeze(3)
        right = x[:, self._scopes[1, :], :].unsqueeze(2)
        result = left + right

        # Put the two channel dimensions from the outer sum into one single dimension:
        # [n, d/2, c, c] -> [n, d/2, c * c]
        result = result.view(*result.shape[:-2], result.shape[-1] ** 2)
        return result

    def __repr__(self):
        return "RatProduct(in_features={}, out_shape={})".format(self.in_features, self.out_shape)


class RatSpn(nn.Module):
    """
    RAT SPN PyTorch implementation with layerwise tensors.

    See also:
    https://arxiv.org/abs/1806.01910
    """

    def __init__(self, region_spns, C, S, I):
        """
        Initialize the RAT SPN  PyTorch module.

        Args:
            region_spns: Internal SPNs which correspond to random region splits.
            C: Number of classes.
            S: Number of sum nodes at each sum layer.
            I: Number of leaf nodes for each leaf region.
        """
        super().__init__()
        self.C = C
        self._priors = nn.Parameter(torch.log(torch.tensor(1 / self.C)), requires_grad=False)
        self.region_spns = nn.ModuleList(region_spns)

        # Root
        in_channels = 0
        for spn in region_spns:
            if spn.num_recursions > 1:
                in_channels += S ** 2
            else:
                in_channels += I ** 2
        self.root = Sum(in_channels=in_channels, in_features=1, out_channels=C)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if hasattr(module, "_init_weights"):
                module._init_weights()
                continue

            if type(module) == Sum:
                truncated_normal_(module.sum_weights, std=0.5)
                continue

    def forward(self, x):
        """Computes the class conditional distributions P(X | C) for each class."""
        # Go over all regions
        split_results = []
        for spn in self.region_spns:
            split_results.append(spn(x).squeeze(1))

        x = torch.stack(split_results, dim=1)

        # Merge results from the different SPN into the channel dimension
        x = x.view(x.shape[0], 1, -1)

        # Apply C sum node outputs
        x = self.root(x)
        x = x.squeeze(1)
        return x


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


if __name__ == "__main__":

    def make_rat(num_features, classes, leaves=10, sums=10, num_splits=10, dropout=0.0):
        rg = RatSpnConstructor(num_features, classes, sums, leaves, dropout=dropout)
        for i in range(2):
            rg.random_split(2, num_splits)
        rat = rg.build()
        return rat

    b = 200
    d = 50
    rat = make_rat(b, 10, num_splits=4)
    import torch

    x = torch.randn(d, b)
    x = rat(x)
    x_norm = torch.logsumexp(x, 1).unsqueeze(1)
    print("x", torch.exp(x - x_norm))
    print("x shape", x.shape)
