import logging
from typing import List, Union

import numpy as np
import torch
from torch import nn

from spn.algorithms.layerwise.distributions import Leaf
from spn.algorithms.layerwise.layers import CrossProduct, Sum
from spn.algorithms.layerwise.utils import provide_evidence
from spn.algorithms.layerwise.type_checks import check_valid
from spn.experiments.RandomSPNs_layerwise.distributions import IndependentMultivariate, RatNormal, truncated_normal_

logger = logging.getLogger(__name__)


class RegionSpn(nn.Module):
    """Defines a single SPN that is create via RatSpnConstructor.random_split(...)."""

    def __init__(
        self,
        S: int,
        I: int,
        dropout: float,
        num_parts: int,
        num_recursions: int,
        in_features: int,
        leaf_base_class: type = RatNormal,
    ):
        """
        Init a Region SPN split.

        Args:
            S (int): Number of sum nodes.
            I (int): Number of distributions for each leaf node.
            dropout (float): Dropout probability.
            num_parts (int): Number of partitions.
            num_recursions (int): Number of split repetitions.
            leaf_base_class (Leaf): 

        """
        super().__init__()

        # Setup class members
        self.S = check_valid(S, int, 1)
        self.I = check_valid(I, int, 1)
        self.dropout = check_valid(dropout, float, 0.0, 1.0)
        self.num_parts = check_valid(num_parts, int, 2, in_features)
        self.num_recursions = check_valid(num_recursions, int, 1, np.log2(in_features))
        self.in_features = check_valid(in_features, int, 1)
        self._leaf_output_features = num_parts ** num_recursions
        self.leaf_base_class = leaf_base_class

        # Build SPN
        self._build()

        # Randomize features
        self.rand_indices = torch.tensor(np.random.permutation(in_features))

    def _build_input_distribution(self):
        # Cardinality is the size of the region in the last partitions
        cardinality = np.ceil(self.in_features / (self.num_parts ** self.num_recursions)).astype(int)
        return IndependentMultivariate(
            multiplicity=self.I,
            in_features=self.in_features,
            cardinality=cardinality,
            dropout=self.dropout,
            leaf_base_class=self.leaf_base_class,
        )

    def _build(self):
        # Build the SPN bottom up:

        # Definition from RAT Paper
        # Leaf Region:      Create I leaf nodes
        # Root Region:      Create C sum nodes
        # Internal Region:  Create S sum nodes
        # Partition:        Cross products of all child-regions

        ### LEAF ###
        self._leaf = self._build_input_distribution()

        # First product layer on top of leaf layer
        prod = CrossProduct(in_features=self.num_parts ** self.num_recursions, in_channels=self.I)
        self._inner_layers = nn.ModuleList()
        self._inner_layers.append(prod)

        # Sum and product layers
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
            sumlayer = Sum(
                in_features=in_features, in_channels=sum_in_channels, out_channels=self.S, dropout=self.dropout
            )

            # Product layer
            prod = CrossProduct(in_features=in_features, in_channels=self.S)

            # Collect
            self._inner_layers.append(sumlayer)
            self._inner_layers.append(prod)

    def forward(self, x: torch.Tensor):
        # Random permutation
        x = x[:, self.rand_indices]

        # Apply leaf distributions
        x = self._leaf(x)

        # Forward to inner product and sum layers
        for l in self._inner_layers:
            x = l(x)

        return x

    def sample(self, n: int = 1, indices: torch.Tensor = None):
        # Sample inner modules
        for l in reversed(self._inner_layers):
            indices = l.sample(n=n, indices=indices)

        # Sample leaf
        samples = self._leaf.sample(n=n, indices=indices)

        # Invert permutation
        inv_rand_indices = invert_permutation(self.rand_indices)
        samples = samples[:, inv_rand_indices]

        return samples

    def _filter_sampling_evidence(self, indices):
        for module in self.modules():
            if isinstance(module, Sum):
                s: Sum = module
                if s._is_sampling_input_cache_enabled:
                    s._sampling_input_cache = s._sampling_input_cache[indices]


def invert_permutation(p: torch.Tensor):
    """
    The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    Taken from: https://stackoverflow.com/a/25535723, adapted to PyTorch.
    """
    s = torch.empty(p.shape[0], dtype=p.dtype, device=p.device)
    s[p] = torch.arange(p.shape[0])
    return s


class RatSpnConstructor:
    def __init__(self, in_features: int, C: int, S: int, I: int, dropout: float = 0.0, leaf_base_class: type = RatNormal):
        """
        RAT SPN class.

        Parameters according to the paper (see Args for descriptions).
        
        Args:
            in_features (int): Number of input features.
            C (int): Number of classes.
            S (int): Number of sum nodes.
            I (int): Number of distributions for each leaf node.
            dropout (float): Dropout probability.
            leaf_base_class (type): Base class for the leaf input distribution.
        """
        self.in_features = check_valid(in_features, int, 1)
        self.C = check_valid(C, int, 1)
        self.S = check_valid(S, int, 1)
        self.I = check_valid(I, int, 1)
        self.dropout = check_valid(dropout, float, 0.0, 1.0)
        assert isinstance(leaf_base_class, type) and issubclass(
            leaf_base_class, Leaf
        ), f"Parameter leaf_base_class must be a subclass type of Leaf but was {leaf_base_class}."
        self.leaf_base_class = leaf_base_class

        # Collect SPNs. Each random_split(...) call adds one SPN
        self._region_spns = []

    def _create_spn(self, num_parts: int, num_recursions: int = 1):
        """Create an SPN from the given RAT parameters.
        
        Args:
            num_parts (int): Number of partitions.
            num_recursions (int, optional): Number of split repetitions. Defaults to 1.
        """
        spn = RegionSpn(self.S, self.I, self.dropout, num_parts, num_recursions, self.in_features, self.leaf_base_class)
        self._region_spns.append(spn)

    def random_split(self, num_parts: int, num_recursions: int = 1):
        """Randomly split the region graph.
        
        Args:
            num_parts (int): Number of partitions.
            num_recursions (int, optional): Number of split repetitions. Defaults to 1.
        """
        if num_parts ** (num_recursions) > self.in_features:
            raise Exception(
                f"The values for num_parts ({num_parts}) and num_recursions ({num_recursions}) have to satisfiy the condition 'num_parts ** num_recursions ({num_parts ** num_recursions}) <= in_features ({self.in_features})'."
            )
        self._create_spn(num_parts, num_recursions)

    def build(self):
        """Build the RatSpn object from the defined region graph"""
        if len(self._region_spns) == 0:
            raise Exception(
                "No random split has been added. Call random_split(...) at least once before building the RatSpn."
            )
        return RatSpn(region_spns=self._region_spns, C=self.C, S=self.S, I=self.I)


class RatSpn(nn.Module):
    """
    RAT SPN PyTorch implementation with layerwise tensors.

    See also:
    https://arxiv.org/abs/1806.01910
    """

    def __init__(self, region_spns: List[RegionSpn], C: int, S: int, I: int):
        """
        Initialize the RAT SPN  PyTorch module.

        Args:
            region_spns: Internal SPNs which correspond to random region splits.
            C: Number of classes.
            S: Number of sum nodes at each sum layer.
            I: Number of leaf nodes for each leaf region.
        """
        super().__init__()
        self.C = check_valid(C, int, 1)
        self.S = check_valid(S, int, 1)
        self.I = check_valid(I, int, 1)
        self.region_spns = nn.ModuleList(region_spns)

        # Compute number of input channels for the root node that stacks all output channels of all RegionSpns
        sum_in_channels = 0
        # Save index interval mapping to region spn index (e.g. index 0-10 belongs to RegionSpn-0, 11-20 to RegionSpn-1, ...)
        # This is helpful bookkeeping for the sampling
        tmp = -1
        self._channels_to_region_spns_intervals = [tmp]
        for spn in region_spns:
            if spn.num_recursions > 1:
                sum_in_channels += S ** 2
                tmp += S ** 2
            else:
                sum_in_channels += I ** 2
                tmp += I ** 2

            self._channels_to_region_spns_intervals.append(tmp)

        self._channels_to_region_spns_intervals[-1] += 1  # Extend last interval bound to make it inclusive

        self.root = Sum(in_channels=sum_in_channels, in_features=1, out_channels=C)

        # Specific root with weights according to priors for sampling
        self._sampling_root = Sum(in_channels=C, in_features=1, out_channels=1)
        self._sampling_root.sum_weights = nn.Parameter(
            torch.ones(size=(1, self.C, 1)) * torch.tensor(1 / self.C).log(), requires_grad=False
        )

        self.init_weights()

    def __device(self):
        """Small hack to obtain the current device."""
        return self._sampling_root.sum_weights.device 

    def init_weights(self):
        for module in self.modules():
            if hasattr(module, "_init_weights"):
                module._init_weights()
                continue

            if isinstance(module, Sum):
                truncated_normal_(module.sum_weights, std=0.5)
                continue

    def forward(self, x: torch.Tensor):
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

    def sample(
        self, n: int = None, class_index: Union[int, List[int]] = None, evidence: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Sample from the RAT-SPN.

        Number of samples is either determined by `n`, number of indices in `class_index` or number of samples in the
        evidence tensor `evidence.shape[0]`.

        Args:
            n (int): Number of samples to generate.
            class_index (Union[int, List[int]]): Sample from a specific class index. Can be either an int, specifying the
                           class, or a list of class indices in which case for each element, a sample of that class will
                           be generated. If not given, sample from root nodes according to uniformly distributed class
                           priors.
            evidence: Sampling evidence.

        Returns:
            torch.Tensor: Samples according to the input.
            
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
            if class_index:
                if isinstance(class_index, list):
                    indices = torch.tensor(class_index, device=self.__device()).view(-1, 1)
                    n = indices.shape[0]
                else:
                    indices = torch.empty(size=(n, 1), device=self.__device())
                    indices.fill_(class_index)
            else:
                # Sample root node (choose one of the classes) TODO: check what happens if C=1
                indices = self._sampling_root.sample(n=n)

            # Sample which region graphs to use
            indices = self.root.sample(indices=indices)

            # Indexes will now point to the stacked channels of all RegionSPN output channels (S*S (if num_recursion > 1)
            # or I*I (else) per SPN). Therefore, we need to bisect the indices

            # Obtain bin assignments
            bin_indices = np.digitize(indices.cpu().numpy(), self._channels_to_region_spns_intervals, right=True)
            bin_indices -= 1  # Offset by -1 since first bins starts with index 1

            # Collect indices for each region spn
            coll = [[] for _ in range(len(self.region_spns))]
            # Collect sample index for each region spn
            evidence_indices = [[] for _ in range(len(self.region_spns))]

            # For each sample
            for i in range(n):
                for ii, b in enumerate(bin_indices[i, :]):
                    # Adjust index according to position in bin
                    # Example, bins: [-1, 3, 8]
                    # Mapping: 0->0, 1->1, 2->2, 3->3,
                    #          4->0, 5->1, 6->2, 7->3
                    # That is, find the lower bound of the current bin, add 1, subtract from actual index to obtain index in that bin
                    # Note: We cannot simply use modulo since bins are not guaranteed to be of the same size!
                    index_adj = indices[i, ii] - (self._channels_to_region_spns_intervals[b] + 1)

                    # Add the adjusted index into the collection for RegionSpn at index 'b'
                    coll[b].append(index_adj)

                    # Save which sample belongs to which region SPN (necessary for sampling with evidence)
                    evidence_indices[b].append(i)

            # Collect all samples
            all_samples = [None for _ in range(n)]

            # Fore each RegionSpn
            for b, (indices, evidence_index) in enumerate(zip(coll, evidence_indices)):

                # Check if some samples were in this bin, else skip
                if len(indices) == 0:
                    continue

                # Convert to torch tensor
                indices = torch.tensor(indices, device=self.__device()).view(-1, 1)

                # Sample b-th RegionSpn with the collected indices
                if evidence is not None:
                    self.region_spns[b]._filter_sampling_evidence(indices=evidence_index)
                samples = self.region_spns[b].sample(indices=indices)

                # Put samples back into correct place
                for i in range(samples.shape[0]):
                    all_samples[evidence_index[i]] = samples[i]

            all_samples = torch.stack(all_samples)
            return all_samples


if __name__ == "__main__":

    def make_rat(num_features, classes, leaves=3, sums=4, num_splits=1, num_recursions=4, dropout=0.0):
        from spn.algorithms.layerwise import distributions as spn_dists

        rg = RatSpnConstructor(num_features, classes, sums, leaves, dropout=dropout, leaf_base_class=spn_dists.Bernoulli)
        for i in range(num_splits):
            rg.random_split(2, num_recursions)
        rat = rg.build()
        return rat

    b = 5
    d = 32
    rat = make_rat(d, 1, num_splits=4)
    import torch

    x = torch.randn(b, d)
    x = rat(x)
    x_norm = torch.logsumexp(x, 1).unsqueeze(1)
    print("x", torch.exp(x - x_norm))
    print("x shape", x.shape)
