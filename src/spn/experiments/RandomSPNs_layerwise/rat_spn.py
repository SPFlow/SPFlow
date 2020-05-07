import logging
from typing import Dict, Type

import numpy as np
import torch
from dataclasses import dataclass
from torch import nn

from spn.algorithms.layerwise.distributions import Leaf
from spn.algorithms.layerwise.layers import CrossProduct, Sum
from spn.algorithms.layerwise.type_checks import check_valid
from spn.algorithms.layerwise.utils import provide_evidence, SamplingContext
from spn.experiments.RandomSPNs_layerwise.distributions import IndependentMultivariate, RatNormal, truncated_normal_

logger = logging.getLogger(__name__)


def invert_permutation(p: torch.Tensor):
    """
    The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    Taken from: https://stackoverflow.com/a/25535723, adapted to PyTorch.
    """
    s = torch.empty(p.shape[0], dtype=p.dtype, device=p.device)
    s[p] = torch.arange(p.shape[0])
    return s


@dataclass
class RatSpnConfig:
    """
    Class for keeping the RatSpn config. Parameter names are according to the original RatSpn paper.

    in_features: int  # Number of input features
    D: int  # Tree depth
    S: int  # Number of sum nodes at each layer
    I: int  # Number of distributions for each scope at the leaf layer
    R: int  # Number of repetitions
    C: int  # Number of root heads / Number of classes
    dropout: float  # Dropout probabilities for leafs and sum layers
    leaf_base_class: Type  # Type of the leaf base class (Normal, Bernoulli, etc)
    leaf_base_kwargs: Dict  # Parameters for the leaf base class
    """

    in_features: int = None
    D: int = None
    S: int = None
    I: int = None
    R: int = None
    C: int = None
    dropout: float = None
    leaf_base_class: Type = None
    leaf_base_kwargs: Dict = None

    @property
    def F(self):
        """Alias for in_features."""
        return self.in_features

    @F.setter
    def F(self, in_features):
        """Alias for in_features."""
        self.in_features = in_features

    def assert_valid(self):
        """Check whether the configuration is valid."""
        self.F = check_valid(self.F, int, 1)
        self.D = check_valid(self.D, int, 1)
        self.C = check_valid(self.C, int, 1)
        self.S = check_valid(self.S, int, 1)
        self.R = check_valid(self.R, int, 1)
        self.I = check_valid(self.I, int, 1)
        self.dropout = check_valid(self.dropout, float, 0.0, 1.0)
        assert self.leaf_base_class is not None, Exception("RatSpnConfig.leaf_base_class parameter was not set!")
        assert isinstance(self.leaf_base_class, type) and issubclass(
            self.leaf_base_class, Leaf
        ), f"Parameter RatSpnConfig.leaf_base_class must be a subclass type of Leaf but was {self.leaf_base_class}."

        if 2 ** self.D > self.F:
            raise Exception(f"The tree depth D={self.D} must be <= {np.floor(np.log2(self.F))} (log2(in_features).")

    def __setattr__(self, key, value):
        if hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"RatSpnConfig object has no attribute {key}")


class RatSpn(nn.Module):
    """
    RAT SPN PyTorch implementation with layer-wise tensors.

    See also:
    https://arxiv.org/abs/1806.01910
    """

    def __init__(self, config: RatSpnConfig):
        """
        Create a RatSpn based on a configuration object.

        Args:
            config (RatSpnConfig): RatSpn configuration object.
        """
        super().__init__()
        config.assert_valid()
        self.config = config

        # Construct the architecture
        self._build()

        # Initialize weights
        self._init_weights()

        # Obtain permutation indices
        self._make_random_repetition_permutation_indices()

    def _make_random_repetition_permutation_indices(self):
        """Create random permutation indices for each repetition."""
        self.rand_indices = torch.empty(size=(self.config.F, self.config.R))
        for r in range(self.config.R):
            # Each repetition has its own randomization
            self.rand_indices[:, r] = torch.tensor(np.random.permutation(self.config.F))

        self.rand_indices = self.rand_indices.long()

    def _randomize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomize the input at each repetition according to `self.rand_indices`.

        Args:
            x: Input.

        Returns:
            torch.Tensor: Randomized input along feature axis. Each repetition has its own permutation.
        """
        # Expand input to the number of repetitions
        x = x.unsqueeze(2)  # Make space for repetition axis
        x = x.repeat((1, 1, self.config.R))  # Repeat R times

        # Random permutation
        for r in range(self.config.R):
            # Get permutation indices for the r-th repetition
            perm_indices = self.rand_indices[:, r]

            # Permute the features of the r-th version of x using the indices
            x[:, :, r] = x[:, perm_indices, r]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RatSpn. Computes the conditional log-likelihood P(X | C).

        Args:
            x: Input.

        Returns:
            torch.Tensor: Conditional log-likelihood P(X | C) of the input.
        """
        # Apply feature randomization for each repetition
        x = self._randomize(x)

        # Apply leaf distributions
        x = self._leaf(x)

        # Pass through intermediate layers
        x = self._forward_layers(x)

        # Merge results from the different repetitions into the channel dimension
        n, d, c, r = x.size()
        assert d == 1  # number of features should be 1 at this point
        x = x.view(n, d, c * r, 1)

        # Apply C sum node outputs
        x = self.root(x)

        # Remove repetition dimension
        x = x.squeeze(3)

        # Remove in_features dimension
        x = x.squeeze(1)

        return x

    def _forward_layers(self, x):
        """
        Forward pass through the inner sum and product layers.

        Args:
            x: Input.

        Returns:
            torch.Tensor: Output of the last layer before the root layer.
        """
        # Forward to inner product and sum layers
        for layer in self._inner_layers:
            x = layer(x)
        return x

    def _build(self):
        """Construct the internal architecture of the RatSpn."""
        # Build the SPN bottom up:
        # Definition from RAT Paper
        # Leaf Region:      Create I leaf nodes
        # Root Region:      Create C sum nodes
        # Internal Region:  Create S sum nodes
        # Partition:        Cross products of all child-regions

        # Construct leaf
        self._leaf = self._build_input_distribution()

        # First product layer on top of leaf layer
        prodlayer = CrossProduct(
            in_features=2 ** self.config.D, in_channels=self.config.I, num_repetitions=self.config.R
        )
        self._inner_layers = nn.ModuleList()
        self._inner_layers.append(prodlayer)

        # Sum and product layers
        sum_in_channels = self.config.I ** 2
        for i in np.arange(start=self.config.D - 1, stop=0, step=-1):
            # Current in_features
            in_features = 2 ** i

            # Sum layer
            sumlayer = Sum(
                in_features=in_features,
                in_channels=sum_in_channels,
                out_channels=self.config.S,
                dropout=self.config.dropout,
                num_repetitions=self.config.R,
            )
            self._inner_layers.append(sumlayer)

            # Product layer
            prodlayer = CrossProduct(in_features=in_features, in_channels=self.config.S, num_repetitions=self.config.R)
            self._inner_layers.append(prodlayer)

            # Update sum_in_channels
            sum_in_channels = self.config.S ** 2

        # Construct root layer
        self.root = Sum(
            in_channels=self.config.R * sum_in_channels, in_features=1, num_repetitions=1, out_channels=self.config.C
        )

        # Construct sampling root with weights according to priors for sampling
        self._sampling_root = Sum(in_channels=self.config.C, in_features=1, out_channels=1, num_repetitions=1)
        self._sampling_root.weights = nn.Parameter(
            torch.ones(size=(1, self.config.C, 1, 1)) * torch.tensor(1 / self.config.C), requires_grad=False
        )

    def _build_input_distribution(self):
        """Construct the input distribution layer."""
        # Cardinality is the size of the region in the last partitions
        cardinality = np.ceil(self.config.F / (2 ** self.config.D)).astype(int)
        return IndependentMultivariate(
            in_features=self.config.F,
            out_channels=self.config.I,
            num_repetitions=self.config.R,
            cardinality=cardinality,
            dropout=self.config.dropout,
            leaf_base_class=self.config.leaf_base_class,
            leaf_base_kwargs=self.config.leaf_base_kwargs,
        )

    @property
    def __device(self):
        """Small hack to obtain the current device."""
        return self._sampling_root.weights.device

    def _init_weights(self):
        """Initiale the weights. Calls `_init_weights` on all modules that have this method."""
        for module in self.modules():
            if hasattr(module, "_init_weights") and module != self:
                module._init_weights()
                continue

            if isinstance(module, Sum):
                truncated_normal_(module.weights, std=0.5)
                continue

    def mpe(self, evidence: torch.Tensor) -> torch.Tensor:
        """
        Perform MPE given some evidence.

        Args:
            evidence: Input evidence. Must contain some NaN values.
        Returns:
            torch.Tensor: Clone of input tensor with NaNs replaced by MPE estimates.
        """
        return self.sample(evidence=evidence, is_mpe=True)

    def sample(self, n: int = None, class_index=None, evidence: torch.Tensor = None, is_mpe: bool = False):
        """
        Sample from the distribution represented by this SPN.

        Possible valid inputs:

        - `n`: Generates `n` samples.
        - `n` and `class_index (int)`: Generates `n` samples from P(X | C = class_index).
        - `class_index (List[int])`: Generates `len(class_index)` samples. Each index `c_i` in `class_index` is mapped
            to a sample from P(X | C = c_i)
        - `evidence`: If evidence is given, samples conditionally and fill NaN values.

        Args:
            n: Number of samples to generate.
            class_index: Class index. Can be either an int in combination with a value for `n` which will result in `n`
                samples from P(X | C = class_index). Or can be a list of ints which will map each index `c_i` in the
                list to a sample from P(X | C = c_i).
            evidence: Evidence that can be provided to condition the samples. If evidence is given, `n` and
                `class_index` must be `None`. Evidence must contain NaN values which will be imputed according to the
                distribution represented by the SPN. The result will contain the evidence and replace all NaNs with the
                sampled values.
            is_mpe: Flag to perform max sampling (MPE).

        Returns:
            torch.Tensor: Samples generated according to the distribution specified by the SPN.

        """
        assert class_index is None or evidence is None, "Cannot provide both, evidence and class indices."
        assert n is None or evidence is None, "Cannot provide both, number of samples to generate (n) and evidence."

        # Check if evidence contains nans
        if evidence is not None:
            assert (evidence != evidence).any(), "Evidence has no NaN values."

            # Set n to the number of samples in the evidence
            n = evidence.shape[0]

        with provide_evidence(self, evidence):  # May be None but that's ok
            # If class is given, use it as base index
            if class_index is not None:
                if isinstance(class_index, list):
                    indices = torch.tensor(class_index, device=self.__device).view(-1, 1)
                    n = indices.shape[0]
                else:
                    indices = torch.empty(size=(n, 1), device=self.__device)
                    indices.fill_(class_index)

                # Create new sampling context
                ctx = SamplingContext(n=n, parent_indices=indices, repetition_indices=None, is_mpe=is_mpe)
            else:
                # Start sampling one of the C root nodes TODO: check what happens if C=1
                ctx = SamplingContext(n=n, is_mpe=is_mpe)
                ctx = self._sampling_root.sample(context=ctx)

            # Sample from RatSpn root layer: Results are indices into the stacked output channels of all repetitions
            ctx.repetition_indices = torch.zeros(n, dtype=int, device=self.__device)
            ctx = self.root.sample(context=ctx)

            # Indexes will now point to the stacked channels of all repetitions (R * S^2 (if D > 1)
            # or R * I^2 (else)).
            root_in_channels = self.root.in_channels // self.config.R
            # Obtain repetition indices
            ctx.repetition_indices = (ctx.parent_indices // root_in_channels).squeeze(1)
            # Shift indices
            ctx.parent_indices = ctx.parent_indices % root_in_channels

            # Now each sample in `indices` belongs to one repetition, index in `repetition_indices`

            # Continue at layers
            # Sample inner layers in reverse order (starting from topmost)
            for layer in reversed(self._inner_layers):
                ctx = layer.sample(context=ctx)

            # Sample leaf
            samples = self._leaf.sample(context=ctx)

            # Invert permutation
            for i in range(n):
                rep_index = ctx.repetition_indices[i]
                inv_rand_indices = invert_permutation(self.rand_indices[:, rep_index])
                samples[i, :] = samples[i, inv_rand_indices]

            if evidence is not None:
                # Update NaN entries in evidence with the sampled values
                nan_indices = torch.isnan(evidence)

                # First make a copy such that the original object is not changed
                evidence = evidence.clone()
                evidence[nan_indices] = samples[nan_indices]
                return evidence
            else:
                return samples
