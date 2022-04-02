import logging
from typing import Dict, Type, List
import math

import numpy as np
import torch as th
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


def invert_permutation(p: th.Tensor):
    """
    The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    Taken from: https://stackoverflow.com/a/25535723, adapted to PyTorch.
    """
    s = th.empty(p.shape[0], dtype=p.dtype, device=p.device)
    s[p] = th.arange(p.shape[0]).to(p.device)
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
    leaf_base_kwargs: Dict  # Parameters for the leaf base class, such as
    #                   tanh_factor: float  # If set, tanh will be applied to samples and taken times this factor.
    gmm_leaves: bool  # If true, the leaves are Gaussian mixtures
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
    gmm_leaves: bool = False
    tanh_squash: bool = False

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
    _inner_layers: nn.ModuleList

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
        permutation = []
        inv_permutation = []
        for r in range(self.config.R):
            permutation.append(th.tensor(np.random.permutation(self.config.F)))
            inv_permutation.append(invert_permutation(permutation[-1]))
        # self.permutation: th.Tensor = th.stack(self.permutation, dim=-1)
        # self.inv_permutation: th.Tensor = th.stack(self.inv_permutation, dim=-1)
        self.permutation = nn.Parameter(th.stack(permutation, dim=-1), requires_grad=False)
        self.inv_permutation = nn.Parameter(th.stack(inv_permutation, dim=-1), requires_grad=False)

    def _randomize(self, x: th.Tensor) -> th.Tensor:
        """
        Randomize the input at each repetition according to `self.permutation`.

        Args:
            x: Input.

        Returns:
            th.Tensor: Randomized input along feature axis. Each repetition has its own permutation.
        """
        # Expand input to the number of repetitions
        n, w = x.shape[:2]
        x = x.unsqueeze(3)  # Make space for repetition axis
        x = x.repeat((1, 1, 1, self.config.R))  # Repeat R times

        # Random permutation
        perm_indices = self.permutation.unsqueeze(0).unsqueeze(0).expand(n, w, -1, -1)
        x = th.gather(x, dim=-2, index=perm_indices)

        return x

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass through RatSpn. Computes the conditional log-likelihood P(X | C).

        Args:
            x: Input of shape [batch, weight_sets, in_features, channel].
                batch: Number of samples per weight set (= per conditional in the CSPN sense).
                weight_sets: In CSPNs, weights are different for each conditional. In RatSpn, this is 1.

        Returns:
            th.Tensor: Conditional log-likelihood P(X | C) of the input.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Apply feature randomization for each repetition
        x = self._randomize(x)

        # Apply leaf distributions
        x = self._leaf(x)

        # Pass through intermediate layers
        x = self._forward_layers(x)

        # Merge results from the different repetitions into the channel dimension
        n, w, d, c, r = x.size()
        assert d == 1  # number of features should be 1 at this point
        # x = th.as_tensor(np.arange(x.shape[-2] * x.shape[-1])).reshape(x.shape[-2], x.shape[-1]).repeat(256, 1, 1, 1)
        # a = th.as_tensor([-10.0, -1.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]).log_softmax(dim=0)
        # a = th.as_tensor([-10.0] * c)
        # a[1] = -1.0
        # a = a.log_softmax(dim=0)
        # x = a.unsqueeze(2).repeat(n, w, d, 1, r)
        x = x.view(n, w, d, c * r, 1)

        # Apply C sum node outputs
        x = self.root(x)

        # Remove repetition dimension
        x = x.squeeze(4)

        # Remove in_features dimension
        x = x.squeeze(2)

        return x

    def _forward_layers(self, x):
        """
        Forward pass through the inner sum and product layers.

        Args:
            x: Input.

        Returns:
            th.Tensor: Output of the last layer before the root layer.
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

        # First product layer on top of leaf layer.
        # May pad output features of leaf layer is their number is not a power of 2.
        prodlayer = CrossProduct(
            in_features=self._leaf.out_features, in_channels=prod_in_channels, num_repetitions=self.config.R
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
            th.ones(size=(1, self.config.C, 1, 1)) * th.tensor(1 / self.config.C), requires_grad=False
        )

    def _build_input_distribution(self, gmm_leaves):
        """Construct the input distribution layer."""
        # Cardinality is the size of the region in the last partitions
        cardinality = np.ceil(self.config.F / (2 ** self.config.D)).astype(int)
        if gmm_leaves:
            return GaussianMixture(in_features=self.config.F, out_channels=self.config.I, gmm_modes=self.config.S,
                                   num_repetitions=self.config.R, cardinality=cardinality, dropout=self.config.dropout,
                                   tanh_squash=self.config.tanh_squash,
                                   leaf_base_class=self.config.leaf_base_class,
                                   leaf_base_kwargs=self.config.leaf_base_kwargs)
        else:
            return IndependentMultivariate(
                in_features=self.config.F, out_channels=self.config.I,
                num_repetitions=self.config.R, cardinality=cardinality, dropout=self.config.dropout,
                tanh_squash=self.config.tanh_squash,
                leaf_base_class=self.config.leaf_base_class,
                leaf_base_kwargs=self.config.leaf_base_kwargs
            )

    @property
    def _device(self):
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

    def mpe(self, evidence: th.Tensor) -> th.Tensor:
        """
        Perform MPE given some evidence.

        Args:
            evidence: Input evidence. Must contain some NaN values.
        Returns:
            th.Tensor: Clone of input tensor with NaNs replaced by MPE estimates.
        """
        return self.sample(evidence=evidence, is_mpe=True)

    def sample(self, mode: str = None, n=1, class_index=None, evidence: th.Tensor = None, is_mpe: bool = False,
               start_at_layer: int = 0):
        """
        Sample from the distribution represented by this SPN.

        Args:
            mode: Two sampling modes are supported:
                'index': Sampling mechanism with indexes, which are non-differentiable.
                'onehot': This sampling mechanism work with one-hot vectors, grouped into tensors.
                          This way of sampling is differentiable, but also takes almost twice as long.
            n: Number of samples to generate.
            class_index: Class index. Can be either an int in combination with a value for `n` which will result in `n`
                samples from P(X | C = class_index). Or can be a list of ints which will map each index `c_i` in the
                list to a sample from P(X | C = c_i).
            evidence: Evidence that can be provided to condition the samples. If evidence is given, `n` and
                `class_index` must be `None`. Evidence must contain NaN values which will be imputed according to the
                distribution represented by the SPN. The result will contain the evidence and replace all NaNs with the
                sampled values.
            is_mpe: Flag to perform max sampling (MPE).
            start_at_layer: Layer to start sampling from. 0 = Root layer, 1 = Child layer of root layer, ...

        Returns:
            th.Tensor: Samples generated according to the distribution specified by the SPN.

        """
        assert mode is not None, "A sampling mode must be provided!"
        assert class_index is None or evidence is None, "Cannot provide both, evidence and class indices."
        assert n is None or evidence is None, "Cannot provide both, number of samples to generate (n) and evidence."

        # Check if evidence contains nans
        if evidence is not None:
            assert (evidence != evidence).any(), "Evidence has no NaN values."

            # Set n to the number of samples in the evidence
            n = evidence.shape[0]

        with provide_evidence(self, evidence, requires_grad=(mode == 'onehot')):  # May be None but that's ok
            # If class is given, use it as base index
            if class_index is not None:
                # Create new sampling context
                ctx = SamplingContext(n=n,
                                      parent_indices=class_index.repeat(n, 1).unsqueeze(-1).to(self._device),
                                      repetition_indices=th.zeros((n, class_index.shape[0]), dtype=int, device=self._device),
                                      is_mpe=is_mpe)
            else:
                # Start sampling one of the C root nodes TODO: check what happens if C=1
                ctx = SamplingContext(n=n, is_mpe=is_mpe)
                # ctx = self._sampling_root.sample(context=ctx)

            if start_at_layer == 0:
                if mode == 'index':
                    # Sample from RatSpn root layer: Results are indices into the
                    # stacked output channels of all repetitions
                    # ctx.repetition_indices = th.zeros(n, dtype=int, device=self._device)
                    ctx = self.root.sample_index_style(ctx=ctx)
                    # parent_indices and repetition indices both have the same shape in the first three dimensions:
                    # [nr_nodes, n, w]
                    # nr_nodes is the number of nodes which are sampled in the current SamplingContext.
                    # In RatSpn.sample() it will always be 1 as we are sampling the root node.
                    # In the variational inference entropy approximation, nr_nodes will be different.
                    # n is the number of samples drawn per node and per weight set.
                    # w is the number of weight sets i.e. the number of conditionals that are given.
                    # This applies only to the Cspn, in the RatSpn this will always be 1.
                else:
                    # Sample from RatSpn root layer: Results are one-hot vectors of the indices
                    # into the stacked output channels of all repetitions
                    ctx = self.root.sample_onehot_style(ctx=ctx)

                # The weights of the root sum node represent the input channel and repetitions in this manner:
                # The CSPN case is assumed where the weights are different for each batch index condition.
                # Looking at one batch index and one output channel, there are IC*R weights.
                # An element of this weight vector is defined as
                # w_{r,c}, with r and c being the repetition and channel the weight belongs to, respectively.
                # The weight vector will then contain [w_{0,0},w_{1,0},w_{2,0},w_{0,1},w_{1,1},w_{2,1},w_{0,2},...]
                # This weight vector was used as the logits in a IC*R-categorical distribution,
                # yielding indexes [0,C*R-1].
                if mode == 'index':
                    # To match the index to the correct repetition and its input channel, we do the following
                    ctx.repetition_indices = (ctx.parent_indices % self.config.R).squeeze(3)
                    ctx.parent_indices = th.div(ctx.parent_indices, self.config.R, rounding_mode='trunc')
                    start_at_layer += 1  # otherwise list index would be out of bounds
                else:
                    assert ctx.parent_indices.shape == (1, ctx.n, self.root.weights.size(0),
                                                        1, self.root.weights.size(2), 1)
                    nr_nodes, n, w, _, _, _ = ctx.parent_indices.shape
                    ctx.parent_indices = ctx.parent_indices.view(nr_nodes, n, w, 1, -1, self.config.R)

            # Continue at layers
            # Sample inner layers in reverse order (starting from topmost)
            # noinspection PyTypeChecker
            for layer in reversed(self._inner_layers[:(len(self._inner_layers)-start_at_layer+1)]):
                if mode == 'index':
                    ctx = layer.sample_index_style(ctx=ctx)
                else:
                    ctx = layer.sample_onehot_style(ctx=ctx)

            if mode == 'onehot':
                assert ctx.parent_indices.shape == (nr_nodes, n, w, self._leaf.out_features,
                                                    self._leaf.out_channels, self.config.R)
            # Sample leaf
            if mode == 'index':
                samples = self._leaf.sample_index_style(ctx=ctx)
            else:
                samples = self._leaf.sample_onehot_style(ctx=ctx)
            if self.config.tanh_squash:
                samples = samples.clamp(-6.0, 6.0).tanh()

            # Invert permutation
            if mode == 'index':
                if ctx.repetition_indices is not None:
                    rep_selected_inv_perm = self.inv_permutation.T[ctx.repetition_indices]
                    samples = th.gather(samples, dim=-1, index=rep_selected_inv_perm)
                else:
                    rep_selected_inv_perm = self.inv_permutation.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    nr_nodes, n, w, f, r = samples.shape
                    rep_selected_inv_perm = rep_selected_inv_perm.expand(nr_nodes, n, w, -1, -1)
                    samples = th.gather(samples, dim=-2, index=rep_selected_inv_perm)
            else:
                rep_selected_inv_permutation = self.inv_permutation * ctx.parent_indices.detach().sum(-2).long()
                rep_selected_inv_permutation = rep_selected_inv_permutation.sum(-1)
                samples = th.gather(samples, dim=-1, index=rep_selected_inv_permutation)

            # The first dimension is the nodes which are sampled. Here, it is always 1 as there is one root node.
            samples.squeeze_(0)

            if evidence is not None:
                # Update NaN entries in evidence with the sampled values
                nan_indices = th.isnan(evidence)

                # First make a copy such that the original object is not changed
                evidence = evidence.clone()
                evidence[nan_indices] = samples[nan_indices]
                return evidence
            else:
                return samples

    def sample_index_style(self, **kwargs):
        return self.sample(mode='index', **kwargs)

    def sample_onehot_style(self, **kwargs):
        return self.sample(mode='onehot', **kwargs)

    @staticmethod
    def calc_aux_responsibility(layer, child_entropies, sample_ll):
        if layer.weights.dim() == 4:
            # RatSpns only have one set of weights, so we must augment the weight_set dimension
            weights = layer.weights.unsqueeze(0)
        else:
            weights = layer.weights
        log_weights = weights.unsqueeze(0)
        weights = log_weights.exp()
        weight_entropy = -(weights * log_weights).sum(dim=3)
        weighted_ch_ents = th.sum(child_entropies.unsqueeze(4) * weights, dim=3)
        aux_resp_ent = log_weights + sample_ll.unsqueeze(4)
        sample_ll = layer(sample_ll)
        aux_resp_ent -= sample_ll.unsqueeze(3)

        # aux_dev_5samples = aux_resp_ent[:5].mean(dim=0, keepdim=True)
        # aux_dev_5samples = th.sum(weights * aux_dev_5samples, dim=3)
        # aux_dev_1sample = aux_resp_ent[:1].mean(dim=0, keepdim=True)
        # aux_dev_1sample = th.sum(weights * aux_dev_1sample, dim=3)

        aux_resp_ent = aux_resp_ent.mean(dim=0, keepdim=True)
        aux_resp_ent = th.sum(weights * aux_resp_ent, dim=3)

        # aux_dev_5samples = (aux_dev_5samples - aux_resp_ent).abs()
        # aux_dev_1sample = (aux_dev_1sample - aux_resp_ent).abs()
        # print(f"Abs. dev. of aux. resp. ent. {aux_resp_ent.mean():.4f} from {aux_resp_ent.shape[0]} samples: "
              # f"5 samples -> {aux_dev_5samples.mean():.4f}, "
              # f"1 sample -> {aux_dev_1sample.mean():.4f}")

        entropy = weight_entropy + weighted_ch_ents + aux_resp_ent
        return entropy, sample_ll

    def vi_entropy_approx(self, sample_size=10, verbose=False, aux_resp_ll_with_grad=False,
                          aux_resp_sample_with_grad=False):
        """
        Approximate the entropy of the root sum node via variational inference,
        as done in the Variational Inference by Policy Search paper.

        Args:
            sample_size: Number of samples to approximate the expected entropy of the responsibility with.
            verbose: Return logging data
            aux_resp_ll_with_grad: When approximating the auxiliary responsibility from log-likelihoods
                of child samples, backpropagate the gradient through the LL calculation.
                This argument will be ignored if this function is called in a th.no_grad() context.
            aux_resp_sample_with_grad: May only be True if aux_resp_ll_with_grad is True too. Backpropagate through
                the sampling of the child nodes as well.
                This argument will be ignored if this function is called in a th.no_grad() context.
        """
        assert not self.config.gmm_leaves, "VI entropy not tested on GMM leaves yet."
        assert self.config.C == 1, "For C > 1, we must calculate starting from self._sampling_root!"
        assert not aux_resp_sample_with_grad or (aux_resp_sample_with_grad and aux_resp_ll_with_grad), \
            "aux_resp_sample_with_grad may only be True if aux_resp_ll_with_grad is True as well."
        root_weights_over_rep = th.empty(1)  # For PyCharm
        log_weights = th.empty(1)
        logging = {}

        child_ll = self._leaf.sample_onehot_style(SamplingContext(n=sample_size, is_mpe=False))
        child_ll = self._leaf(child_ll)
        child_entropies = -child_ll.mean(dim=0, keepdim=True)

        for i in range(len(self._inner_layers) + 1):
            if i < len(self._inner_layers):
                layer = self._inner_layers[i]
            else:
                layer = self.root

            if isinstance(layer, CrossProduct):
                child_entropies = layer(child_entropies)
            else:
                with th.set_grad_enabled(aux_resp_ll_with_grad and th.is_grad_enabled()):
                    ctx = SamplingContext(n=sample_size, is_mpe=False)
                    if aux_resp_sample_with_grad and th.is_grad_enabled():
                        # noinspection PyTypeChecker
                        for child_layer in reversed(self._inner_layers[:i]):
                            ctx = child_layer.sample_onehot_style(ctx)
                        child_sample = self._leaf.sample_onehot_style(ctx)
                    else:
                        with th.no_grad():
                            # noinspection PyTypeChecker
                            for child_layer in reversed(self._inner_layers[:i]):
                                ctx = child_layer.sample_index_style(ctx)
                            child_sample = self._leaf.sample_index_style(ctx)

                    # The nr_nodes is the number of input channels (ic) to the
                    # current layer - we sampled all its input channels.
                    ic, n, w, d, r = child_sample.shape

                    # Combine first two dims of child_ll.
                    # child_ll [0,0] -> [0], ..., [0, n-1] -> [n-1], [1, 0] -> [n], ...
                    child_sample = child_sample.view(ic * n, w, d, r)
                    child_ll = self._leaf(child_sample)
                    _, w, d, leaf_oc, r = child_ll.shape
                    child_ll = child_ll.view(ic, n, w, d, leaf_oc, r)

                    # We can average over the sample_size dimension with size 'n' here already.
                    child_ll = child_ll.mean(dim=1)

                    # noinspection PyTypeChecker
                    for child_layer in self._inner_layers[:i]:
                        child_ll = child_layer(child_ll)

                    if i == len(self._inner_layers):
                        # Now we are dealing with a log-likelihood tensor with the shape [ic, w, 1, ic, r],
                        # where child_ll[0,0,:,:,0] are the log-likelihoods of the ic nodes in the first repetition
                        # given the samples from the first node of that repetition.
                        # The problem is that the weights of the root sum node don't recognize different, separated
                        # repetitions, so we reshape the weights to make the repetition dimension explicit again.
                        # This is equivalent to splitting up the root sum node into one sum node per repetition,
                        # with another sum node sitting on top.
                        root_weights_over_rep = layer.weights.view(w, 1, r, ic).permute(0, 1, 3, 2).unsqueeze(-2)
                        log_weights = th.log_softmax(root_weights_over_rep, dim=2)
                        # child_ll and the weights are log-scaled, so we add them together.
                        ll = child_ll.unsqueeze(-2) + log_weights.unsqueeze(0)
                        # ll shape [ic, w, 1, ic, r]
                        ll = th.logsumexp(ll, dim=3)

                        # first reshape the tensor to get the nodes over which we sampled into
                        # the first dimension.
                        # ll = child_ll.permute(0, 4, 1, 2, 3).reshape(ic * r, w, 1, ic)
                        # ll[0,0,:,:] are the log-likelihoods of the samples from the first node computed among the 'ic'
                        # other nodes within that repetition. But the root sum nodes wants the log-likelihoods of the
                        # other 'ic*(r-1)' nodes w.r.t. to that same sample as well! They are all zero.

                    else:
                        ll = layer(child_ll)

                        # We have the log-likelihood of the current sum layer w.r.t. the samples from its children.
                        # We permute the dims so this tensor is of shape [w, d, ic, oc, r]
                    ll = ll.permute(1, 2, 0, 3, 4)

                    # child_ll now contains the log-likelihood of the samples from all of its 'ic' nodes per feature and
                    # repetition - ic * d * r in total.
                    # child_ll contains the LL of the samples of each node evaluated among all other nodes - separated
                    # by repetition and feature.
                    # The tensor shape is [ic, w, d, ic, r]. Looking at one weight set, one feature and one repetition,
                    # we are looking at the slice [:, 0, 0, :, 0].
                    # The first dimension is the dimension of the samples - there are 'ic' of them.
                    # The 4th dimension is the dimension of the LLs of the nodes for those samples.
                    # So [4, 0, 0, :, 0] contains the LLs of all nodes given the sample from the fifth node.
                    # Likewise, [:, 0, 0, 2, 0] contains the LLs of the samples of all nodes, evaluated at the third node.
                    # We needed the full child_ll tensor to compute the LLs of the current layer, but now we only
                    # require the LLs of each node's own samples.
                    child_ll = child_ll[range(ic), :, :, range(ic), :]  # [ic, w, d, r]
                    child_ll = child_ll.permute(1, 2, 0, 3)
                    child_ll = child_ll.unsqueeze(3)  # [w, d, ic, 1, r]

                weight_entropy = -(layer.weights.exp() * layer.weights).sum(dim=2)
                if not i == len(self._inner_layers):
                    log_weights = layer.weights
                weights = log_weights.exp()
                child_entropies.squeeze_(0)
                weighted_ch_ents = th.sum(child_entropies.unsqueeze(3) * weights, dim=2)
                aux_responsibility = log_weights.detach() + child_ll - ll
                weighted_aux_responsibility = th.sum(weights * aux_responsibility, dim=2)
                if i == len(self._inner_layers):
                    weight_entropy = weight_entropy.squeeze(-1)
                    weights = root_weights_over_rep.exp().sum(dim=2).softmax(dim=-1)
                    weighted_ch_ents = th.sum(weighted_ch_ents * weights, dim=-1)
                    weighted_aux_responsibility = th.sum(weights * weighted_aux_responsibility, dim=-1)
                child_entropies = weight_entropy + weighted_ch_ents + weighted_aux_responsibility
                child_entropies.unsqueeze_(0)
                if verbose:
                    weight_entropy = -(layer.weights.exp() * layer.weights).sum(dim=2)
                    metrics = {
                        'weight_entropy': weight_entropy.detach(),
                        'weighted_child_ent': weighted_ch_ents.detach(),
                        'weighted_aux_resp': weighted_aux_responsibility.detach(),
                    }
                    logging[i] = {}
                    for rep in range(weight_entropy.size(-1)):
                        rep_key = f"rep{rep}"
                        rep = th.as_tensor(rep, device=self._device)
                        for key, metric in metrics.items():
                            logging[i].update({
                                f"{rep_key}/{key}/min": metric.index_select(-1, rep).min().item(),
                                f"{rep_key}/{key}/max": metric.index_select(-1, rep).max().item(),
                                f"{rep_key}/{key}/mean": metric.index_select(-1, rep).mean().item(),
                                f"{rep_key}/{key}/std": metric.index_select(-1, rep).std(dim=0).mean().item(),
                            })

        return child_entropies.flatten(), logging

    def old_vi_entropy_approx(self, sample_size):
        assert not self.config.gmm_leaves, "VI entropy not tested on GMM leaves yet."
        # To calculate the entropies layer by layer, starting from the leaves.
        # We repurpose the samples of the leaves when moving up through the layers.
        ctx = SamplingContext(n=sample_size, is_mpe=False)
        sample = self._leaf.sample(ctx)
        sample_ll = self._leaf(sample)
        del sample

        # Apply permutation
        # sample_ll = self._randomize(sample_ll)
        child_entropies = -sample_ll.mean(dim=0, keepdim=True)
        # deviation_5samples = (child_entropies + sample_ll[:5].mean(dim=0, keepdim=True)).abs()
        # deviation_1sample = (child_entropies + sample_ll[:1].mean(dim=0, keepdim=True)).abs()
        # print(f"Abs. dev. of child_ents {child_entropies.mean():.4f} from {sample_size} samples: "
              # f"5 samples -> {deviation_5samples.mean():.4f}, "
              # f"1 sample -> {deviation_1sample.mean():.4f}")
        for layer in self._inner_layers:
            if isinstance(layer, CrossProduct):
                child_entropies = layer(child_entropies)
                sample_ll = layer(sample_ll)
            else:
                child_entropies, sample_ll = self.calc_aux_responsibility(layer, child_entropies, sample_ll)

        n, w, d, c, r = sample_ll.size()
        assert d == 1  # number of features should be 1 at this point
        child_entropies = child_entropies.view(1, w, d, c * r, 1)
        sample_ll = sample_ll.view(n, w, d, c * r, 1)
        child_entropies, sample_ll = self.calc_aux_responsibility(self.root, child_entropies, sample_ll)
        child_entropies = child_entropies.flatten()
        return child_entropies

    def consolidate_weights(self):
        """
            This function calculates the weights of the network if it were a hierarchical mixture model,
            that is, without product layers. These weights are needed for calculating the entropy.
        """
        current_weights: th.Tensor = self.root.weights
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

    def consolidated_vector_forward(self, leaf_vectors: List[th.Tensor], kernel) -> List[th.Tensor]:
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
            inv_rand_indices = invert_permutation(self.permutation[:, i])
            for g in leaf_vectors:
                g[:, :, :, :, i] = g[:, :, inv_rand_indices, :, i]
        leaf_vectors = kernel(leaf_vectors, self._sampling_root)
        leaf_vectors = [v.sum(-1).squeeze_(1).squeeze_(-1) for v in leaf_vectors]
        # each tensor in leaf_vectors has shape [n, 1, self.config.F, self.config.C, 1]
        return leaf_vectors

    @staticmethod
    def weighted_sum_kernel(child_grads: th.Tensor, layer: Sum):
        weights = layer.consolidated_weights.unsqueeze(2)
        # Weights is of shape [n, d, 1, ic, oc, r]
        # The extra dimension is created so all elements of the gradient vectors are multiplied by the same
        # weight for that feature and output channel.
        return [(g.unsqueeze(4) * weights).sum(dim=3) for g in child_grads]

    @staticmethod
    def moment_kernel(child_moments: List[th.Tensor], layer: Sum):
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

    def compute_gradients(self, x: th.Tensor, with_log_prob_x=False, order=3):
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
                log_sum_weights: th.Tensor = layer.weights
                if log_sum_weights.dim() == 4:
                    # Only in the Cspn case are the weights already log-normalized
                    log_sum_weights: th.Tensor = th.log_softmax(log_sum_weights, dim=2)
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
        inner_sum_ent = th.cat(inner_sum_ent, dim=0)
        norm_inner_sum_ent = th.cat(norm_inner_sum_ent, dim=0)
        log_root_weights = self.root.weights
        if log_root_weights.dim() == 4:
            # Only in the Cspn case are the weights already log-normalized
            log_root_weights: th.Tensor = th.log_softmax(log_root_weights, dim=2)
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
