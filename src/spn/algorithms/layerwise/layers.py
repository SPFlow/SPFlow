import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from spn.algorithms.layerwise.type_checks import check_valid
from spn.algorithms.layerwise.utils import provide_evidence

logger = logging.getLogger(__name__)


class AbstractLayer(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, n: int = 1, indices: torch.Tensor = None) -> torch.Tensor:
        """
        Sample from this layer.
        Args:
            n: Number of samples.
            indices: Parent indices.

        Returns:
            torch.Tensor: Generated samples.
        """
        pass


class Sum(AbstractLayer):
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

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_sampling_input_cache_enabled = False
        self._sampling_input_cache = None

    def _enable_sampling_input_cache(self):
        """Enables the input cache. This will store the input in forward passes into `self.__input_cache`."""
        self._is_sampling_input_cache_enabled = True

    def _disable_sampling_input_cache(self):
        """Disables and clears the input cache."""
        self._is_sampling_input_cache_enabled = False
        self._sampling_input_cache = None

    def forward(self, x: torch.Tensor):
        """
        Sum layer foward pass.

        Args:
            x: Input of shape [batch, in_features, in_channels].

        Returns:
            torch.Tensor: Output of shape [batch, in_features, out_channels]
        """
        # Save input if input cache is enabled
        if self._is_sampling_input_cache_enabled:
            self._sampling_input_cache = x.clone()

        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_indices] = np.NINF

        # Multiply x with weights in log-space
        # Results in shape: [n, d, ic, oc]
        x = x.unsqueeze(3) + F.log_softmax(self.sum_weights, dim=1)

        # Compute sum via logsumexp along dimension "ic" (in_channels)
        # Results in shape: [n, d, oc]
        x = torch.logsumexp(x, dim=2)

        return x

    def sample(self, n: int = 1, indices: torch.Tensor = None) -> torch.Tensor:
        """Method to sample from this layer, based on the parents output.

        Output is always a vector of indices into the channels.

        Args:
            indices (torch.Tensor): Parent sampling output.
            n (int): Number of samples.
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
        """

        # Sum weights are of shape: in_features x in_channels x out_channels
        # We now want to use `indices` to acces one channel for each in_feature x out_channels block
        # index is of size in_feature

        sum_weights = self.sum_weights.data

        # If this is not the root node, use the paths (out channels), specified by the parent layer
        if indices is not None:
            sum_weights = sum_weights[range(self.in_features), :, indices]

            # Apply log_softmax to ensure they are proper probabilities
            sum_weights = F.log_softmax(sum_weights, dim=2)
        else:
            assert sum_weights.shape[2] == 1, "Cannot start sampling from non-root layer"
            sum_weights = sum_weights[:, :, [0] * n]

            # Apply log_softmax to ensure they are proper probabilities
            sum_weights = F.log_softmax(sum_weights, dim=1)

            # Move sample dimension to the first axis: [feat, channels, batch] -> [batch, feat, channels]
            sum_weights = sum_weights.permute(2, 0, 1)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._sampling_input_cache is not None:
            sum_weights.mul_(self._sampling_input_cache)

        # Create categorical distribution and use weights as logits
        dist = torch.distributions.Categorical(logits=sum_weights)
        indices = dist.sample()
        return indices

    def __repr__(self):
        return "Sum(in_channels={}, in_features={}, out_channels={}, dropout={}, out_shape={})".format(
            self.in_channels, self.in_features, self.out_channels, self.dropout, self.out_shape
        )


class Product(AbstractLayer):
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

    def sample(self, n: int = 1, indices: torch.Tensor = None) -> torch.Tensor:
        """Method to sample from this layer, based on the parents output.

        Args:
            n (int): Number of instances to sample.
            indices (torch.Tensor): Parent sampling output.
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """

        # If this is a root node
        if indices is None:
            # TODO: check if this is correct?
            return torch.zeros(n, 1)

        # Repeat the parent indices, e.g. [0, 2, 3] -> [0, 0, 2, 2, 3, 3]
        # depending on the cardinality
        sample_indices = torch.repeat_interleave(indices, repeats=self.cardinality)
        sample_indices = sample_indices.view(indices.shape[0], -1)

        # Remove padding
        if self._pad:  # TODO: test if padding works along broadcasting
            sample_indices = sample_indices[:, : -self._pad]

        return sample_indices

    def __repr__(self):
        return "Product(in_features={}, cardinality={}, out_shape={})".format(
            self.in_features, self.cardinality, self.out_shape
        )


class CrossProduct(AbstractLayer):
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
        self.in_features = check_valid(in_features, int, 1)
        self.in_channels = check_valid(in_channels, int, 1)
        cardinality = 2  # Fixed to binary graphs for now
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
        self.unraveled_channel_indices = nn.Parameter(torch.tensor(
            [(i, j) for i in range(self.in_channels) for j in range(self.in_channels)]
        ), requires_grad=False)

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

    def sample(self, n: int = 1, indices: torch.Tensor = None) -> torch.Tensor:
        """Method to sample from this layer, based on the parents output.

        Args:
            n: Number of samples.
            indices (torch.Tensor): Parent sampling output
        Returns:
            torch.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """

        # If this is a root node
        if indices is None:
            # TODO: check if this is correct?
            return torch.zeros(n, 1)

        # Map flattened indexes back to coordinates to obtain the chosen input_channel for each feature
        indices = self.unraveled_channel_indices[indices]
        indices = indices.view(indices.shape[0], -1)

        # Remove padding
        if self._pad:  # TODO: test if padding works along broadcasting
            indices = indices[:, : -self._pad]

        return indices

    def __repr__(self):
        return "CrossProduct(in_features={}, out_shape={})".format(self.in_features, self.out_shape)

