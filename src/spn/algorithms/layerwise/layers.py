import logging
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class Sum(nn.Module):
    def __init__(self, in_channels, in_features, out_channels, dropout=0.0):
        """
        Create a Sum layer.

        Args:
            in_channels (int): Number of output channels from the previous layer.
            in_features (int): Number of input features.
            out_channels (int): Multiplicity of a sum node for a given scope set.
            dropout (float, optional): Dropout percentage.
        """
        super(Sum, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.out_channels = out_channels
        self.dropout = dropout
        assert out_channels > 0, (
            "Number of output channels must be at least 1, but was %s." % out_channels
        )
        in_features = int(in_features)
        # Weights, such that each sumnode has its own weights
        ws = torch.randn(in_features, in_channels, out_channels)
        self.sum_weights = nn.Parameter(ws)
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=dropout)

        self.out_shape = f"(N, {in_features}, C_in)"

    def forward(self, x):
        """
        Sum layer foward pass.

        Args:
            x: Input of shape [batch, in_features, in_channels].

        Returns:
            torch.Tensor: Output of shape [batch, in_features, out_channels]
        """
        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        if self.dropout > 0.0:
            r = self._bernoulli_dist.sample(x.shape).type(torch.bool)
            x[r] = np.NINF

        # Multiply x with weights in logspace
        # Resuts in shape: [n, d, ic, oc]
        x = x.unsqueeze(3) + F.log_softmax(self.sum_weights, dim=1)

        # Compute sum via logsumexp along dimension "ic" (in_channels)
        # Resuts in shape: [n, d, oc]
        x = torch.logsumexp(x, dim=2)

        return x

    def __repr__(self):
        return "Sum(in_channels={}, in_features={}, out_channels={}, dropout={}, out_shape={})".format(
            self.in_channels,
            self.in_features,
            self.out_channels,
            self.dropout,
            self.out_shape,
        )


class Product(nn.Module):
    """
    Product Node Layer that chooses k scopes as children for a product node.
    """

    def __init__(self, in_features, cardinality):
        """
        Create a product node layer.

        Args:
            in_features (int): Number of input features.
            cardinality (int): Number of random children for each product node.
        """

        super(Product, self).__init__()
        self.in_features = in_features
        self.cardinality = int(cardinality)

        in_features = int(in_features)
        self._out_features = np.ceil(in_features / cardinality).astype(int)
        self.out_shape = f"(N, {self._out_features}, C_in)"

    def forward(self, x):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [batch, in_features, channel].

        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/cardinality), channel].
        """
        # Only one product node
        if self.cardinality == x.shape[1]:
            return x.sum(1)

        x_split = list(torch.split(x, self.cardinality, dim=1))

        # Check if splits have the same shape (If split cannot be made even, the last chunk will be smaller)
        if x_split[-1].shape != x_split[0].shape:
            # How much is the last chunk smaller
            diff = x_split[0].shape[1] - x_split[-1].shape[1]

            # Pad the last chunk by the difference with zeros (=maginalized nodes)
            x_split[-1] = F.pad(
                x_split[-1], pad=[0, 0, 0, diff], mode="constant", value=0.0
            )

        # Stack along new split axis
        x_split_stack = torch.stack(x_split, dim=2)

        # Sum over feature axis
        result = torch.sum(x_split_stack, dim=1)
        return result

    def __repr__(self):
        return "Product(in_features={}, cardinality={}, out_shape={})".format(
            self.in_features, self.cardinality, self.out_shape
        )
