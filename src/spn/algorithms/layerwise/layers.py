import logging
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Union
from spn.algorithms.layerwise.type_checks import check_valid
from spn.algorithms.layerwise.distributions import Leaf


logger = logging.getLogger(__name__)


class Sum(nn.Module):
    def __init__(self, in_channels: int, in_features: int, out_channels: int, dropout: float = 0.0):
        """
        Create a Sum layer.

        Args:
            in_channels (int): Number of output channels from the previous layer.
            in_features (int): Number of input features.
            out_channels (int): Multiplicity of a sum node for a given scope set.
            dropout (float, optional): Dropout percentage.
        """
        super(Sum, self).__init__()

        self.in_channels = check_valid(in_channels, int, 1)
        self.in_features = check_valid(in_features, int, 1)
        self.out_channels = check_valid(out_channels, int, 1)
        self.dropout = nn.Parameter(torch.tensor(check_valid(dropout, float, 0.0, 1.0)), requires_grad=False)

        # Weights, such that each sumnode has its own weights
        ws = torch.randn(self.in_features, self.in_channels, self.out_channels)
        self.sum_weights = nn.Parameter(ws)
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=self.dropout)

        self.out_shape = f"(N, {self.in_features}, {self.out_channels})"

    def forward(self, x: torch.Tensor):
        """
        Sum layer foward pass.

        Args:
            x: Input of shape [batch, in_features, in_channels].

        Returns:
            torch.Tensor: Output of shape [batch, in_features, out_channels]
        """
        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        if self.dropout > 0.0 and self.training:
            dropout_idxs = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_idxs] = np.NINF

        # Multiply x with weights in logspace
        # Resuts in shape: [n, d, ic, oc]
        x = x.unsqueeze(3) + F.log_softmax(self.sum_weights, dim=1)

        # Compute sum via logsumexp along dimension "ic" (in_channels)
        # Resuts in shape: [n, d, oc]
        x = torch.logsumexp(x, dim=2)

        return x

    def sample(self, idxs: torch.Tensor = None, n: int = 1) -> torch.Tensor:
        """Method to sample from this layer, based on the parents output.

        Output is always a vector of indices into the channels.

        Args:
            idxs (torch.Tensor): Parent sampling output.
            n (int): Number of samples.
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
        """

        # Sum weights are of shape: in_features x in_channels x out_channels
        # We now want to use `idxs` to acces one channel for each in_feature x out_channels block
        # idx is of size in_feature

        sum_weights = self.sum_weights

        # If this is not the root node, use the paths (out channels), specified by the parent layer
        if idxs is not None:
            sum_weights = sum_weights[range(self.in_features), :, idxs]

            # Apply log_softmax to ensure they are proper probabilities
            sum_weights = F.log_softmax(sum_weights, dim=2)
        else:
            assert sum_weights.shape[2] == 1, "Cannot start sampling from non-root layer"
            sum_weights = sum_weights[:, :, [0] * n]

            # Apply log_softmax to ensure they are proper probabilities
            sum_weights = F.log_softmax(sum_weights, dim=1)

            # Move sample dimension to the first axis: [feat, channels, batch] -> [batch, feat, channels]
            sum_weights = sum_weights.permute(2, 0, 1)

        # Permute sum weights such that dim0: in_features, dim1: out_channels, dim2: in_channels
        # since Categorial uses the last channel as axis for the probabilities and we want to
        # sample from the input channels and use their weights as probabilities
        # sum_weights_permuted = sum_weights_softmaxed.permute(0, 2, 1)

        # Create categorical distribution and use weights as logits
        dist = torch.distributions.Categorical(logits=sum_weights)
        indices = dist.sample()
        return indices

    def __repr__(self):
        return "Sum(in_channels={}, in_features={}, out_channels={}, dropout={}, out_shape={})".format(
            self.in_channels, self.in_features, self.out_channels, self.dropout, self.out_shape
        )


class Product(nn.Module):
    """
    Product Node Layer that chooses k scopes as children for a product node.
    """

    def __init__(self, in_features: int, cardinality: int):
        """
        Create a product node layer.

        Args:
            in_features (int): Number of input features.
            cardinality (int): Number of random children for each product node.
        """

        super(Product, self).__init__()

        self.in_features = check_valid(in_features, int, 1)
        self.cardinality = check_valid(cardinality, int, 2, in_features + 1)

        # Implement product as convolution
        self._conv_weights = nn.Parameter(torch.ones(1, 1, cardinality, 1), requires_grad=False)
        self._pad = (self.cardinality - self.in_features % self.cardinality) % self.cardinality

        self._out_features = np.ceil(self.in_features / self.cardinality).astype(int)
        self.out_shape = f"(N, {self._out_features}, C_in)"

    def forward(self, x: torch.Tensor):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [batch, in_features, channel].

        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/cardinality), channel].
        """
        # Only one product node
        if self.cardinality == x.shape[1]:
            return x.sum(1, keepdim=True)

        # Pad if in_features % cardinality != 0
        if self._pad > 0:
            x = F.pad(x, pad=(0, 0, 0, self._pad), value=0)

        # Use convolution with weights of 1 and stride/kernel size of #children
        # Simulate a single feature map, therefore [n, d, c] -> [n, c'=1, d, c], that is
        # - The convolution channel input and output size will be 1
        # - The feature dimension (d) will be the height dimension
        # - The channel dimension (c) will be the width dimension
        # Convolution is then applied along the width with stride/ksize #children and along
        # the height with stride/ksize 1
        x = x.unsqueeze(1)
        result = F.conv2d(x, weight=self._conv_weights, stride=(self.cardinality, 1))

        # Remove simulated channel
        result = result.squeeze(1)
        return result

    def sample(self, idxs: torch.Tensor = None, n: int = None) -> torch.Tensor:
        """Method to sample from this layer, based on the parents output.

        Args:
            x (torch.Tensor): Parent sampling output
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """

        # If this is a root node
        if idxs is None:
            # TODO: check if this is correct?
            return torch.zeros(n, 1)

        # Repeat the parent indices, e.g. [0, 2, 3] -> [0, 0, 2, 2, 3, 3]
        # depending on the cardinality
        sample_idxs = torch.repeat_interleave(idxs, repeats=self.cardinality)
        sample_idxs = sample_idxs.view(idxs.shape[0], -1)

        # Remove padding
        if self._pad:  # TODO: test if padding works along broadcasting
            sample_idxs = sample_idxs[:, : -self._pad]

        return sample_idxs

    def __repr__(self):
        return "Product(in_features={}, cardinality={}, out_shape={})".format(
            self.in_features, self.cardinality, self.out_shape
        )


class CrossProduct(nn.Module):
    """
    Layerwise implementation of a RAT Product node.

    Builds the the combination of all children in two regions:
    res = []
    for n1 in R1, n2 in R2:
        res += [n1 * n2]
        
    TODO: Generalize to k regions (cardinality = k).
    """

    def __init__(self, in_features: int, in_channels: int):
        """
        Create a rat product node layer.

        Args:
            in_features (int): Number of input features.
            in_channels (int): Number of input channels. This is only needed for the sampling pass.
        """

        super().__init__()
        cardinality = 2  # Fixed to binary graphs for now
        self.in_features = check_valid(in_features, int, 1)
        self.in_channels = check_valid(in_channels, int, 1)
        self.cardinality = check_valid(cardinality, int, 2, in_features + 1)
        self._out_features = np.ceil(self.in_features / self.cardinality).astype(int)
        self._pad = 0

        # Collect scopes for each product child
        self._scopes = [[] for _ in range(self.cardinality)]

        # Create sequence of scopes
        scopes = np.arange(self.in_features)

        # For two consecutive scopes
        for i in range(0, self.in_features, self.cardinality):
            for j in range(cardinality):
                if i + j < in_features:
                    self._scopes[j].append(scopes[i + j])
                else:
                    # Case: d mod cardinality != 0 => Create marginalized nodes with prob 1.0
                    # Pad x in forward pass on the right: [n, d, c] -> [n, d+1, c] where index
                    # d+1 is the marginalized node (index "in_features")
                    self._scopes[j].append(self.in_features)

        # Transform into numpy array for easier indexing
        self._scopes = np.array(self._scopes)

        # Create index map from flattened to coordinates (only needed in sampling)
        self.unraveled_channel_indices = torch.tensor(
            [(i, j) for i in range(self.in_channels) for j in range(self.in_channels)]
        )

        self.out_shape = f"(N, {self._out_features}, {self.in_channels ** 2})"

    def forward(self, x: torch.Tensor):
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
            self._pad = 2 ** np.ceil(np.log2(x.shape[1])).astype(np.int) - x.shape[1]

            # Pad marginalized node
            x = F.pad(x, pad=[0, 0, 0, self._pad], mode="constant", value=0.0)

        # Build outer sum, using broadcasting, this can be done with
        # modifying the tensor dimensions:
        # left: [n, d/2, c] -> [n, d/2, c, 1]
        # right: [n, d/2, c] -> [n, d/2, 1, c]
        # left + right with broadcasting: [n, d/2, c, 1] + [n, d/2, 1, c] -> [n, d/2, c, c]
        left = x[:, self._scopes[0, :], :].unsqueeze(3)
        right = x[:, self._scopes[1, :], :].unsqueeze(2)
        result = left + right

        # Put the two channel dimensions from the outer sum into one single dimension:
        # [n, d/2, c, c] -> [n, d/2, c * c]
        N, D2, C, C = result.shape
        # result = result.view(*result.shape[:-2], result.shape[-1] ** 2)
        result = result.view(N, D2, C * C)
        return result

    def sample(self, idxs: torch.Tensor = None, n: int = None) -> torch.Tensor:
        """Method to sample from this layer, based on the parents output.

        Args:
            x (torch.Tensor): Parent sampling output
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """

        # If this is a root node
        if idxs is None:
            # TODO: check if this is correct?
            return torch.zeros(n, 1)

        # Map flattened indexes back to coordinates to obtain the chosen input_channel for each feature
        # idxs_old = idxs
        idxs = self.unraveled_channel_indices[idxs]
        idxs = idxs.view(idxs.shape[0], -1)

        # Remove padding
        if self._pad:  # TODO: test if padding works along broadcasting
            idxs = idxs[:, : -self._pad]

        return idxs

    def __repr__(self):
        return "CrossProduct(in_features={}, out_shape={})".format(self.in_features, self.out_shape)


class StackedSPN(nn.Module):
    """A class that encapsulates the hierarchy of an SPN using the layerwise implementation."""

    def __init__(self):
        super().__init__()
        self._leaf = None
        self._layers = nn.ModuleList()

    @property
    def leaf(self):
        return self._leaf

    @leaf.setter
    def leaf(self, layer: Leaf):
        assert isinstance(layer, Leaf)
        self._leaf = leaf

    def add_layer(self, layer: Union[Sum, Product, CrossProduct, Leaf]):
        assert isinstance(layer, (Sum, Product, CrossProduct))

        self._layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward to leaf
        x = self._leaf(x)

        # Forward to inner product and sum layers
        for l in self._layers:
            x = l(x)

        return x

    def sample(self, idxs: torch.Tensor, n: int = 1):
        # Sample inner modules
        for l in reversed(self._layers):
            idxs = l.sample(idxs, n)

        # Sample leaf
        samples = self._leaf.sample(idxs, n)

        return samples


if __name__ == "__main__":
    root_sum = Sum(in_channels=3, in_features=4, out_channels=1)
    x = torch.randn(10, 3)

    root_sum.sample()
