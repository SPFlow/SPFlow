import logging
from typing import Dict, Type, List
import math

import numpy as np
import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from torch import distributions as dist

from base_distributions import Leaf
from layers import CrossProduct, Sum
from type_checks import check_valid
from utils import provide_evidence, SamplingContext
from distributions import IndependentMultivariate, GaussianMixture, truncated_normal_

logger = logging.getLogger(__name__)


def invert_permutation(p: torch.Tensor):
    """
    The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    Taken from: https://stackoverflow.com/a/25535723, adapted to PyTorch.
    """
    s = torch.empty(p.shape[0], dtype=p.dtype, device=p.device)
    s[p] = torch.arange(p.shape[0]).to(p.device)
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
    gmm_leaves: bool = True

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
        # x = torch.as_tensor(np.arange(x.shape[-2] * x.shape[-1])).reshape(x.shape[-2], x.shape[-1]).repeat(256, 1, 1, 1)
        # a = torch.as_tensor([-10.0, -1.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]).log_softmax(dim=0)
        # a = torch.as_tensor([-10.0] * c)
        # a[1] = -1.0
        # a = a.log_softmax(dim=0)
        # x = a.unsqueeze(1).repeat(n, d, 1, r)
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
        self._leaf = self._build_input_distribution(gmm_leaves=self.config.gmm_leaves)

        self._inner_layers = nn.ModuleList()
        prod_in_channels = self.config.I

        # First product layer on top of leaf layer
        prodlayer = CrossProduct(
            in_features=2 ** self.config.D, in_channels=prod_in_channels, num_repetitions=self.config.R
        )
        self._inner_layers.append(prodlayer)
        sum_in_channels = self.config.I ** 2

        # Sum and product layers
        for i in np.arange(start=self.config.D - 1, stop=0, step=-1):
            # Current in_features
            in_features = 2 ** i

            # Sum layer
            sumlayer = Sum(in_features=in_features, in_channels=sum_in_channels, num_repetitions=self.config.R,
                           out_channels=self.config.S, dropout=self.config.dropout)
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

    def _build_input_distribution(self, gmm_leaves):
        """Construct the input distribution layer."""
        # Cardinality is the size of the region in the last partitions
        cardinality = np.ceil(self.config.F / (2 ** self.config.D)).astype(int)
        if gmm_leaves:
            return GaussianMixture(in_features=self.config.F, out_channels=self.config.I, gmm_modes=self.config.S,
                                   num_repetitions=self.config.R, cardinality=cardinality, dropout=self.config.dropout,
                                   leaf_base_class=self.config.leaf_base_class,
                                   leaf_base_kwargs=self.config.leaf_base_kwargs)
        else:
            return IndependentMultivariate(in_features=self.config.F, out_channels=self.config.I,
                                   num_repetitions=self.config.R, cardinality=cardinality, dropout=self.config.dropout,
                                   leaf_base_class=self.config.leaf_base_class,
                                   leaf_base_kwargs=self.config.leaf_base_kwargs)

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

    def sample(self, n: int = None, class_index=None, evidence: torch.Tensor = None, is_mpe: bool = False, **kwargs):
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

            # The weights of the root sum node represent the input channel and repetitions in this manner:
            # The CSPN case is assumed where the weights are different for each batch index condition.
            # Looking at one batch index and one output channel, there are IC*R weights.
            # An element of this weight vector is defined as
            # w_{r,c}, with r and c being the repetition and channel the weight belongs to, respectively.
            # The weight vector will then contain [w_{0,0},w_{1,0},w_{2,0},w_{0,1},w_{1,1},w_{2,1},w_{0,2},w_{1,2},...]
            # This weight vector was used as the logits in a IC*R-categorical distribution, yielding indexes [0,C*R-1].
            # To match the index to the correct repetition and its input channel, we do the following
            ctx.repetition_indices = (ctx.parent_indices % self.config.R).squeeze(1)
            ctx.parent_indices = torch.div(ctx.parent_indices, self.config.R, rounding_mode='trunc')

            if kwargs.get('override_root'):
                a = np.arange(self.root.in_channels // self.config.R)
                b = np.arange(self.config.R)
                a = torch.as_tensor(a).to(self.__device)
                b = torch.as_tensor(b).to(self.__device)
                a = a.repeat(self.config.R)
                b = b.repeat_interleave(self.root.in_channels // self.config.R)
                ctx.parent_indices = a.unsqueeze(1)
                ctx.repetition_indices = b

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

    def consolidate_weights(self):
        """
            This function calculates the weights of the network if it were a hierarchical mixture model,
            that is, without product layers. These weights are needed for calculating the entropy.
        """
        current_weights: torch.Tensor = self.root.weights
        assert current_weights.dim() == 5, "This isn't adopted to the 4-dimensional RatSpn weights yet"

        n, d, ic, oc, _ = current_weights.shape
        # root mean weights have shape [n, 1, S^2*R, C, 1]
        # The sampling root weights the repetitions
        assert oc == 1, "Check if the sampling root weights are calculated correctly for C>1."

        s_root_weights = current_weights.softmax(dim=2).view(n, d, ic // self.config.R, oc, self.config.R)
        s_root_weights = s_root_weights.sum(dim=2, keepdim=True)
        self._sampling_root.consolidated_weights = s_root_weights
        # The weights in the root are reshaped to account for the repetitions,
        # so the product layer can make use of them.
        current_weights = current_weights.view(n, d, ic // self.config.R, oc, self.config.R).softmax(dim=2)
        current_sum: Sum = self.root
        for layer in reversed(self._inner_layers):
            if isinstance(layer, CrossProduct):
                current_weights = layer.consolidate_weights(parent_weights=current_weights)
                current_sum.consolidated_weights = current_weights
            else:  # Is a sum layer
                current_sum: Sum = layer
                current_weights = layer.weights.softmax(dim=2)

    def consolidated_vector_forward(self, leaf_vectors: List[torch.Tensor], kernel) -> List[torch.Tensor]:
        """
            Performs an upward pass on vectors from the leaf layer. Such vectors have a length of 'cardinality'.
            The upward pass calls 'kernel' at each sum node.
            The results are given into a weighted sum, with the weights being the consolidated weights of the Sum.
            At each product layer, the vectors are concatenated, making them twice as long and halving the number
            of features.
        """
        cardinality = self._leaf.cardinality
        out_channels = self._leaf.out_channels
        r = self.config.R
        n = self._leaf.base_leaf.means.shape[0]

        features = self._leaf.out_features
        if np.log2(features) % 1 != 0.0:
            pad = 2 ** np.ceil(np.log2(features)).astype(np.int) - features
            leaf_vectors = [F.pad(g, pad=[0, 0, 0, 0, 0, 0, 0, pad], mode="constant", value=g.mean().item())
                            if g is not None else None
                            for g in leaf_vectors]
            features += pad
        for layer in self._inner_layers:
            if isinstance(layer, Sum):
                leaf_vectors = kernel(leaf_vectors, layer)
                out_channels = layer.out_channels
            else:
                if layer.in_features != features:
                    # Concatenate grad vectors together, as the features now decrease in number
                    leaf_vectors = [g.view(n, layer.in_features, cardinality * 2, out_channels, r)
                                    for g in leaf_vectors]
                    features = layer.in_features
                    cardinality *= 2
        leaf_vectors = kernel(leaf_vectors, self.root)
        leaf_vectors = [g.view(n, 1, -1, self.config.C, self.config.R)
                        for g in leaf_vectors]
        if leaf_vectors[0].size(2) != self.config.F:
            leaf_vectors = [g[:, :, :self.config.F] for g in leaf_vectors]
        for i in range(self.config.R):
            inv_rand_indices = invert_permutation(self.rand_indices[:, i])
            for g in leaf_vectors:
                g[:, :, :, :, i] = g[:, :, inv_rand_indices, :, i]
        leaf_vectors = kernel(leaf_vectors, self._sampling_root)
        leaf_vectors = [v.sum(-1).squeeze_(1).squeeze_(-1) for v in leaf_vectors]
        # each tensor in leaf_vectors has shape [n, 1, self.config.F, self.config.C, 1]
        return leaf_vectors

    @staticmethod
    def weighted_sum_kernel(child_grads: torch.Tensor, layer: Sum):
        weights = layer.consolidated_weights.unsqueeze(2)
        # Weights is of shape [n, d, 1, ic, oc, r]
        # The extra dimension is created so all elements of the gradient vectors are multiplied by the same
        # weight for that feature and output channel.
        return [(g.unsqueeze(4) * weights).sum(dim=3) for g in child_grads]

    @staticmethod
    def moment_kernel(child_moments: List[torch.Tensor], layer: Sum):
        assert layer.consolidated_weights is not None, "No consolidated weights are set for this Sum node!"
        weights = layer.consolidated_weights.unsqueeze(2)
        # Weights is of shape [n, d, 1, ic, oc, r]
        # Create an extra dimension for the mean vector so all elements of the mean vector are multiplied by the same
        # weight for that feature and output channel.

        child_mean = child_moments[0]
        # moments have shape [n, d, cardinality, ic, r]
        # Create an extra 'output channels' dimension, as the weights are separate for each output channel.
        child_mean.unsqueeze_(4)
        mean = child_mean * weights
        # mean has shape [n, d, cardinality, ic, oc, r]
        mean = mean.sum(dim=3)
        # mean has shape [n, d, cardinality, oc, r]
        moments = [mean]

        centered_mean = child_var = 0
        if len(child_moments) >= 2:
            child_var = child_moments[1]
            child_var.unsqueeze_(4)
            centered_mean = child_mean - mean.unsqueeze(4)
            var = child_var + centered_mean**2
            var = var * weights
            var = var.sum(dim=3)
            moments += [var]

        if len(child_moments) >= 3:
            child_skew = child_moments[2]
            skew = 3 * centered_mean * child_var + centered_mean ** 3
            if child_skew is not None:
                child_skew.unsqueeze_(4)
                skew = skew + child_skew
            skew = skew * weights
            skew = skew.sum(dim=3)
            moments += [skew]

        # layer.mean, layer.var, layer.skew = mean, var, skew
        return moments

    def compute_moments(self, order=3):
        moments = self._leaf.moments()
        if len(moments) < order:
            moments += [None] * (order - len(moments))
        return self.consolidated_vector_forward(moments, RatSpn.moment_kernel)

    def compute_gradients(self, x: torch.Tensor, with_log_prob_x=False, order=3):
        x = self._randomize(x)
        grads: List = self._leaf.gradient(x, order=order)
        if len(grads) < order:
            grads += [None] * (order - len(grads))
        if with_log_prob_x:
            log_p = self._leaf(x, reduction=None)
            grads += [log_p]
        return self.consolidated_vector_forward(grads, RatSpn.weighted_sum_kernel)

    def sum_node_entropies(self, reduction='mean'):
        inner_sum_ent = []
        norm_inner_sum_ent = []
        for i in range(1, len(self._inner_layers)):
            layer = self._inner_layers[i]
            if isinstance(layer, Sum):
                log_sum_weights: torch.Tensor = layer.weights
                if log_sum_weights.dim() == 4:
                    # Only in the Cspn case are the weights already log-normalized
                    log_sum_weights: torch.Tensor = torch.log_softmax(log_sum_weights, dim=2)
                assert self.sum.weights.dim() == 5, "This isn't adopted to the 4-dimensional RatSpn weights yet"
                nr_cat = log_sum_weights.shape[2]
                max_categ_ent = -np.log(1/nr_cat)
                categ_ent = -(log_sum_weights.exp() * log_sum_weights).sum(dim=2)
                norm_categ_ent = categ_ent / max_categ_ent
                if reduction == 'mean':
                    categ_ent = categ_ent.mean()
                    norm_categ_ent = norm_categ_ent.mean()
                inner_sum_ent.append(categ_ent.unsqueeze(0))
                norm_inner_sum_ent.append(norm_categ_ent.unsqueeze(0))
        inner_sum_ent = torch.cat(inner_sum_ent, dim=0)
        norm_inner_sum_ent = torch.cat(norm_inner_sum_ent, dim=0)
        log_root_weights = self.root.weights
        if log_root_weights.dim() == 4:
            # Only in the Cspn case are the weights already log-normalized
            log_root_weights: torch.Tensor = torch.log_softmax(log_root_weights, dim=2)
        nr_cat = log_root_weights.shape[2]
        max_categ_ent = -np.log(1 / nr_cat)
        root_categ_ent = -(log_root_weights.exp() * log_root_weights).sum(dim=2)
        norm_root_categ_ent = root_categ_ent / max_categ_ent
        if reduction == 'mean':
            inner_sum_ent = inner_sum_ent.mean()
            norm_inner_sum_ent = norm_inner_sum_ent.mean()
            root_categ_ent = root_categ_ent.mean()
            norm_root_categ_ent = norm_root_categ_ent.mean()

        return inner_sum_ent, norm_inner_sum_ent, root_categ_ent, norm_root_categ_ent
