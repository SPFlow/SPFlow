import logging
from abc import ABC, abstractmethod
from typing import List, Union, Tuple

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
import torch.distributions as dist

from type_checks import check_valid
from utils import SamplingContext

logger = logging.getLogger(__name__)


class AbstractLayer(nn.Module, ABC):
    def __init__(self, in_features: int, num_repetitions: int = 1):
        super().__init__()
        self.in_features = check_valid(in_features, int, 1)
        self.num_repetitions = check_valid(num_repetitions, int, 1)

    @abstractmethod
    def sample(self, ctx: SamplingContext = None) -> Union[SamplingContext, th.Tensor]:
        """
        Sample from this layer.
        Args:
            ctx: Sampling context.

        Returns:
            th.Tensor: Generated samples.
        """
        pass

    @abstractmethod
    def sample_index_style(self, ctx: SamplingContext = None) -> Union[SamplingContext, th.Tensor]:
        pass

    @abstractmethod
    def sample_onehot_style(self, ctx: SamplingContext = None) -> Union[SamplingContext, th.Tensor]:
        pass


class Sum(AbstractLayer):
    def __init__(
        self, in_channels: int, in_features: int, out_channels: int, num_repetitions: int = 1, dropout: float = 0.0
    ):
        """
        Create a Sum layer.

        Input is expected to be of shape [n, d, ic, r].
        Output will be of shape [n, d, oc, r].

        Args:
            in_channels (int): Number of output channels from the previous layer.
            in_features (int): Number of input features.
            out_channels (int): Multiplicity of a sum node for a given scope set.
            num_repetitions(int): Number of layer repetitions in parallel.
            dropout (float, optional): Dropout percentage.
        """
        super().__init__(in_features, num_repetitions)

        self.in_channels = check_valid(in_channels, int, 1)
        self.out_channels = check_valid(out_channels, int, 1)
        self.dropout = nn.Parameter(th.tensor(check_valid(dropout, float, 0.0, 1.0)), requires_grad=False)

        # Weights, such that each sumnode has its own weights
        ws = th.randn(self.in_features, self.in_channels, self.out_channels, self.num_repetitions)
        self.weights = nn.Parameter(ws)
        self._bernoulli_dist = th.distributions.Bernoulli(probs=self.dropout)

        self.out_shape = f"(N, {self.in_features}, {self.out_channels}, {self.num_repetitions})"

        # Necessary for sampling with evidence: Save input during forward pass.
        self._is_input_cache_enabled = False
        self._input_cache = None

        # Weights of this sum node that were propagated down through the following product layer.
        # These weights weigh the child sum node in the layer after the product layer that follows this one.
        # The consolidated weights are needed for moment calculation.
        self.consolidated_weights = None
        self.mean = None
        self.var = None
        self.skew = None

    def _enable_input_cache(self):
        """Enables the input cache. This will store the input in forward passes into `self.__input_cache`."""
        self._is_input_cache_enabled = True

    def _disable_input_cache(self):
        """Disables and clears the input cache."""
        self._is_input_cache_enabled = False
        self._input_cache = None

    @property
    def __device(self):
        """Hack to obtain the current device, this layer lives on."""
        return self.weights.device

    def forward(self, x: th.Tensor):
        """
        Sum layer forward pass.

        Args:
            x: Input of shape [batch, weight_sets, in_features, in_channels].
                weight_sets: In CSPNs, there are separate weights for each batch element.

        Returns:
            th.Tensor: Output of shape [batch, in_features, out_channels]
        """
        # Save input if input cache is enabled
        if self._is_input_cache_enabled:
            self._input_cache = x.clone()

        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        if self.dropout > 0.0 and self.training:
            dropout_indices = self._bernoulli_dist.sample(x.shape).bool()
            x[dropout_indices] = np.NINF

        # Dimensions
        n, w, d, ic, r = x.size()
        x = x.unsqueeze(4)  # Shape: [n, w, d, ic, 1, r]
        if self.weights.dim() == 4:
            # RatSpns only have one set of weights, so we must augment the weight_set dimension
            weights = self.weights.unsqueeze(0)
        else:
            weights = self.weights
        # Weights is of shape [n, d, ic, oc, r]
        oc = weights.size(3)
        # The weights must be expanded by the batch dimension so all samples of one conditional see the same weights.
        log_weights = weights.unsqueeze(0)

        # Multiply (add in log-space) input features and weights
        x = x + log_weights  # Shape: [n, w, d, ic, oc, r]

        # Compute sum via logsumexp along in_channels dimension
        x = th.logsumexp(x, dim=3)  # Shape: [n, w, d, oc, r]

        # Assert correct dimensions
        assert x.size() == (n, w, d, oc, r)

        return x

    def sample(self, ctx: SamplingContext = None) -> Union[SamplingContext, th.Tensor]:
        raise NotImplementedError("sample() has been split up into sample_index_style() and sample_onehot_style()!"
                                  "Please choose one.")

    def sample_index_style(self, ctx: SamplingContext = None) -> SamplingContext:
        """Method to sample from this layer, based on the parents output.

        Output is always a vector of indices into the channels.

        Args:
            ctx: Contains
                repetition_indices (List[int]): An index into the repetition axis for each sample.
                    Can be None if `num_repetitions==1`.
                parent_indices (th.Tensor): Parent sampling output.
                n: Number of samples to draw for each set of weights.
        Returns:
            th.Tensor: Index into tensor which paths should be followed.
        """

        # Sum weights are of shape: [N, D, IC, OC, R]
        # We now want to use `indices` to access one in_channel for each in_feature x out_channels block
        # index is of size in_feature
        weights: th.Tensor = self.weights
        if weights.dim() == 4:
            weights = weights.unsqueeze(0)
        # w is the number of weight sets
        w, d, ic, oc, r = weights.shape
        sample_size = ctx.n

        # Create sampling context if this is a root layer
        if ctx.is_root:
            weights = weights.unsqueeze(0).expand(sample_size, -1, -1, -1, -1, -1)
            # weights from selected repetition with shape [n, w, d, ic, oc, r]
            # In this sum layer there are oc * r nodes per feature. oc * r is our nr_nodes.
            weights = weights.permute(5, 4, 0, 1, 2, 3)
            # weights from selected repetition with shape [r, oc, n, w, d, ic]
            # Reshape weights to [oc * r, n, w, d, ic]
            # The nodes in the first dimension are taken from the first two weight dimensions [r, oc] like this:
            # [0, 0], ..., [0, oc-1], [1, 0], ..., [1, oc-1], [2, 0], ..., [r-1, oc-1]
            # This means the weights for the first oc nodes are the weights for repetition 0.
            # This must coincide with the repetition indices.
            weights = weights.reshape(oc * r, sample_size, w, d, ic)

            ctx.repetition_indices = th.arange(r).to(self.__device).repeat_interleave(oc)
            ctx.repetition_indices = ctx.repetition_indices.unsqueeze(-1).unsqueeze(-1).repeat(
                1, ctx.n, w
            )

        else:
            # If this is not the root node, use the paths (out channels), specified by the parent layer
            if ctx.repetition_indices is not None:
                self._check_repetition_indices(ctx)

            weights = weights.expand(ctx.parent_indices.shape[0], sample_size, -1, -1, -1, -1, -1)
            parent_indices = ctx.parent_indices.unsqueeze(4).unsqueeze(4)
            if ctx.repetition_indices is not None:
                rep_ind = ctx.repetition_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                rep_ind = rep_ind.expand(-1, -1, -1, d, ic, oc, -1)
                weights = th.gather(weights, dim=-1, index=rep_ind).squeeze(-1)
                # weights from selected repetition with shape [nr_nodes, n, w, d, ic, oc]
                parent_indices = parent_indices.expand(-1, -1, -1, -1, ic, -1)
            else:
                parent_indices = parent_indices.expand(-1, -1, -1, -1, ic, -1, -1)
            weights = th.gather(weights, dim=5, index=parent_indices).squeeze(5)
            # weights from selected parent with shape [nr_nodes, n, w, d, ic]

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache is not None:
            raise NotImplementedError("Not yet adapted to new sampling method")
            for i in range(w):
                # Reweight the i-th samples weights by its likelihood values at the correct repetition
                weights[i, :, :] += self._input_cache[i, :, :, ctx.repetition_indices[i]]

        # If sampling context is MPE, set max weight to 1 and rest to zero, such that the maximum index will be sampled
        if ctx.is_mpe:
            # Get index of largest weight along in-channel dimension
            indices = weights.argmax(dim=4)
        else:
            # Create categorical distribution and use weights as logits.
            #
            # Use the Gumble-Softmax trick to obtain one-hot indices of the categorical distribution
            # represented by the given logits. (Use Gumble-Softmax instead of Categorical
            # to allow for gradients).
            #
            # The code below is an approximation of:
            #
            # >> dist = th.distributions.Categorical(logits=weights)
            # >> indices = dist.sample()

            one_hot = F.gumbel_softmax(logits=weights, hard=True, dim=4)
            cats = th.arange(ic, device=weights.device)
            if weights.dim() == 6:
                cats = cats.unsqueeze_(-1).expand(-1, r)
            indices = (one_hot * cats).sum(4).long()

        # Update parent indices
        ctx.parent_indices = indices

        return ctx

    def sample_onehot_style(self, ctx: SamplingContext = None) -> SamplingContext:
        """Method to sample from this layer, based on the parents output.

        Output is always a one-hot vector of indices into the channels.

        Args:
            ctx: Contains
                repetition_indices (List[int]): An index into the repetition axis for each sample.
                    Can be None if `num_repetitions==1`.
                parent_indices (th.Tensor): Parent sampling output.
                n: Number of samples to draw for each set of weights.
        Returns:
            th.Tensor: Index into tensor which paths should be followed.
        """

        # Sum weights are of shape: [N, D, IC, OC, R]
        # We now want to use the "hot" indices to access one in_channel for each in_feature x out_channels block
        weights: th.Tensor = self.weights
        if weights.dim() == 4:
            weights = weights.unsqueeze(0)
        # w is the number of weight sets
        w, d, ic, oc, r = weights.shape
        sample_size = ctx.n

        # Create sampling context if this is a root layer
        if ctx.is_root:
            weights = weights.unsqueeze(0).expand(sample_size, -1, -1, -1, -1, -1)
            # Shape [n, w, d, ic, oc, r]
            weights = weights.permute(4, 0, 1, 2, 3, 5)
            # oc is our nr_nodes: [nr_nodes, n, w, d, ic, r]
        else:
            assert ctx.parent_indices.detach().sum(4).max().item() == 1.0
            weights = weights * ctx.parent_indices.unsqueeze(4)
            # [nr_nodes, n, w, d, ic, oc, r]
            weights = weights.sum(5)  # Sum out output_channel dimension
            if ctx.parent_indices.detach()[0, 0, 0, 0, :, :].sum().item() == 1.0:
                # Only one repetition is selected, remove repetition dim of weights
                weights = weights.sum(-1)

        # If evidence is given, adjust the weights with the likelihoods of the observed paths
        if self._is_input_cache_enabled and self._input_cache is not None:
            raise NotImplementedError("Not yet adapted to new sampling method")
            for i in range(w):
                # Reweight the i-th samples weights by its likelihood values at the correct repetition
                weights[i, :, :] += self._input_cache[i, :, :, ctx.repetition_indices[i]]

        # If sampling context is MPE, set max weight to 1 and rest to zero, such that the maximum index will be sampled
        if ctx.is_mpe:
            # Get index of largest weight along in-channel dimension
            indices = weights.argmax(dim=4)
            one_hot = F.one_hot(indices, num_classes=ic)
            if one_hot.dim() == 6:
                # F.one_hot() expands the last dim, which is the ic dim. It must come before the repetition dim.
                one_hot = one_hot.permute(0, 1, 2, 3, 5, 4)
        else:
            # Create categorical distribution and use weights as logits.
            #
            # Use the Gumble-Softmax trick to obtain one-hot indices of the categorical distribution
            # represented by the given logits. (Use Gumble-Softmax instead of Categorical
            # to allow for gradients).
            #
            # The code below is an approximation of:
            #
            # >> dist = th.distributions.Categorical(logits=weights)
            # >> indices = dist.sample()

            one_hot = F.gumbel_softmax(logits=weights, hard=True, dim=4)

        if one_hot.dim() == 5:
            # Weights didn't have repetition dim, so re-instantiate it again.
            one_hot = ctx.parent_indices.detach().sum(4).unsqueeze(4) * one_hot.unsqueeze(-1)
        assert one_hot.detach().sum(4).max().item() == 1.0

        # Update one-hot vectors to select the input channels of this layer.
        ctx.parent_indices = one_hot

        return ctx

    def depr_forward_grad(self, child_grads):
        weights = self.consolidated_weights.unsqueeze(2)
        return [(g.unsqueeze_(4) * weights).sum(dim=3) for g in child_grads]

    def depr_compute_moments(self, child_moments: List[th.Tensor]):
        assert self.consolidated_weights is not None, "No consolidated weights are set for this Sum node!"
        # Create an extra dimension for the mean vector so all elements of the mean vector are multiplied by the same
        # weight for that feature and output channel.
        weights = self.consolidated_weights.unsqueeze(2)
        # Weights is of shape [n, d, 1, ic, oc, r]

        mean, var, skew = [m.unsqueeze(4) if m is not None else None for m in child_moments]
        # moments have shape [n, d, cardinality, ic, r]
        # Create an extra 'output channels' dimension, as the weights are separate for each output channel.
        self._mean = mean * weights
        # _mean has shape [n, d, cardinality, ic, oc, r]
        self._mean = self._mean.sum(dim=3)
        # _mean has shape [n, d, cardinality, oc, r]

        centered_mean = mean - self._mean.unsqueeze(4)
        self._var = var + centered_mean**2
        self._var = self._var * weights
        self._var = self._var.sum(dim=3)

        self._skew = 3*centered_mean*var + centered_mean**3
        if skew is not None:
            self._skew = self._skew + skew
        self._skew = self._skew * weights
        self._skew = self._skew.sum(dim=3)

        return self._mean, self._var, self._skew

    def _check_repetition_indices(self, ctx: SamplingContext):
        assert ctx.repetition_indices.shape[0] == ctx.parent_indices.shape[0]
        assert ctx.repetition_indices.shape[1] == ctx.parent_indices.shape[1]
        if self.num_repetitions > 1 and ctx.repetition_indices is None:
            raise Exception(
                f"Sum layer has multiple repetitions (num_repetitions=={self.num_repetitions}) but repetition_indices argument was None, expected a Long tensor size #samples."
            )
        if self.num_repetitions == 1 and ctx.repetition_indices is None:
            ctx.repetition_indices = th.zeros(ctx.n, dtype=int, device=self.__device)

    def __repr__(self):
        return "Sum(in_channels={}, in_features={}, out_channels={}, dropout={}, out_shape={})".format(
            self.in_channels, self.in_features, self.out_channels, self.dropout, self.out_shape
        )


class Product(AbstractLayer):
    """
    Product Node Layer that chooses k scopes as children for a product node.
    """

    def __init__(self, in_features: int, cardinality: int, num_repetitions: int = 1):
        """
        Create a product node layer.

        Args:
            in_features (int): Number of input features.
            cardinality (int): Number of random children for each product node.
        """

        super().__init__(in_features, num_repetitions)

        self.cardinality = check_valid(cardinality, int, 1, in_features + 1)

        # Implement product as convolution
        # self._conv_weights = nn.Parameter(th.ones(1, 1, cardinality, 1, 1), requires_grad=False)
        self._pad = (self.cardinality - self.in_features % self.cardinality) % self.cardinality

        self._out_features = np.ceil(self.in_features / self.cardinality).astype(int)
        self.out_shape = f"(N, {self._out_features}, in_channels, {self.num_repetitions})"

    # @property
    # def __device(self):
        # """Hack to obtain the current device, this layer lives on."""
        # return self._conv_weights.device

    def forward(self, x: th.Tensor, reduction = 'sum'):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [batch, weight_sets, in_features, channel, repetitions].

        Returns:
            th.Tensor: Output of shape [batch, ceil(in_features/cardinality), channel].
        """
        # Only one product node
        if self.cardinality == x.shape[2]:
            if reduction == 'sum':
                return x.sum(2, keepdim=True)
            else:
                return x

        # Special case: if cardinality is 1 (one child per product node), this is a no-op
        if self.cardinality == 1:
            return x

        # Pad if in_features % cardinality != 0
        if self._pad > 0:
            x = F.pad(x, pad=(0, 0, 0, 0, 0, self._pad), value=0)

        # Dimensions
        n, w, d, c, r = x.size()
        d_out = d // self.cardinality

        if reduction is None:
            return x.view(n, w, d_out, self.cardinality, c, r)
        elif reduction == 'sum':
            x = x.view(n * w, d, c, r)
            x = F.conv3d(x.unsqueeze(1), weight=th.ones(1, 1, self.cardinality, 1, 1, device=x.device),
                         stride=(self.cardinality, 1, 1))
            return x.view(n, w, d_out, c, r)
        else:
            raise NotImplementedError("No reduction other than sum is implemented. ")

    def sample_onehot_style(self, ctx: SamplingContext = None) -> Union[SamplingContext, th.Tensor]:
        return self.sample(ctx)

    def sample_index_style(self, ctx: SamplingContext = None) -> Union[SamplingContext, th.Tensor]:
        return self.sample(ctx)

    def sample(self, n: int = None, ctx: SamplingContext = None) -> SamplingContext:
        """Method to sample from this layer, based on the parents output.

        Args:
            n (int): Number of instances to sample.
            indices (th.Tensor): Parent sampling output.
        Returns:
            th.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """

        # If this is a root node
        if ctx.is_root:

            if self.num_repetitions == 1:
                # If there is only a single repetition, create new sampling context
                ctx.parent_indices = th.zeros(ctx.n, 1, dtype=int, device=self.__device)
                ctx.repetition_indices = th.zeros(ctx.n, dtype=int, device=self.__device)
                return ctx
            else:
                raise Exception(
                    "Cannot start sampling from Product layer with num_repetitions > 1 and no context given."
                )
        else:
            # Repeat the parent indices, e.g. [0, 2, 3] -> [0, 0, 2, 2, 3, 3] depending on the cardinality
            indices = th.repeat_interleave(ctx.parent_indices, repeats=self.cardinality, dim=3)

            # Remove padding
            if self._pad:
                indices = indices[:, :, :, : -self._pad]

            ctx.parent_indices = indices
            return ctx

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

    def __init__(self, in_features: int, in_channels: int, num_repetitions: int = 1):
        """
        Create a rat product node layer.

        Args:
            in_features (int): Number of input features.
            in_channels (int): Number of input channels. This is only needed for the sampling pass.
        """

        # Check if padding to next power of 2 is necessary
        self._pad = 2 ** np.ceil(np.log2(in_features)).astype(np.int) - in_features
        super().__init__(2 ** np.ceil(np.log2(in_features)).astype(np.int), num_repetitions)

        self.in_channels = check_valid(in_channels, int, 1)
        cardinality = 2  # Fixed to binary graphs for now
        self.cardinality = check_valid(cardinality, int, 2, in_features + 1)
        self._out_features = np.ceil(self.in_features / self.cardinality).astype(int)

        # Collect scopes for each product child
        self._scopes = [[] for _ in range(self.cardinality)]

        # Create sequence of scopes
        scopes = np.arange(self.in_features)

        # For two consecutive scopes
        for i in range(0, self.in_features, self.cardinality):
            for j in range(cardinality):
                self._scopes[j].append(scopes[i + j])
                # if i + j < in_features:
                    # self._scopes[j].append(scopes[i + j])
                # else:
                    # Case: d mod cardinality != 0 => Create marginalized nodes with prob 1.0
                    # Pad x in forward pass on the right: [n, d, c] -> [n, d+1, c] where index
                    # d+1 is the marginalized node (index "in_features")
                    # self._scopes[j].append(self.in_features)

        # Transform into numpy array for easier indexing
        self._scopes = np.array(self._scopes)

        # Create index map from flattened to coordinates (only needed in sampling)
        self.unraveled_channel_indices = nn.Parameter(
            th.tensor([(i, j) for i in range(self.in_channels)
                          for j in range(self.in_channels)]),
            requires_grad=False
        )
        self.one_hot_in_channel_mapping = nn.Parameter(F.one_hot(self.unraveled_channel_indices).float(),
                                                       requires_grad=False)
        # Number of conditionals (= number of different weight sets) in the CSPN.
        # This is only needed when sampling this layer as root.
        # It is initialized as 1, which would also be the RatSpn case.
        # It is only set in the CSPN.set_weights() function.
        self.num_conditionals = 1

        self.out_shape = f"(N, {self._out_features}, {self.in_channels ** 2}, {self.num_repetitions})"

    @property
    def __device(self):
        """Hack to obtain the current device, this layer lives on."""
        return self.unraveled_channel_indices.device

    def forward(self, x: th.Tensor):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [batch, weight_sets, in_features, channel].
                weight_sets: In CSPNs, there are separate weights for each batch element.

        Returns:
            th.Tensor: Output of shape [batch, ceil(in_features/2), channel * channel].
        """
        if self._pad:
            # Pad marginalized node
            x = F.pad(x, pad=[0, 0, 0, 0, 0, self._pad], mode="constant", value=0.0)

        # Dimensions
        n, w, d, c, r = x.size()
        d_out = d // self.cardinality

        # Build outer sum, using broadcasting, this can be done with
        # modifying the tensor dimensions:
        # left: [n, d/2, c, r] -> [n, d/2, c, 1, r]
        # right: [n, d/2, c, r] -> [n, d/2, 1, c, r]
        left = x[:, :, self._scopes[0, :], :, :].unsqueeze(4)
        right = x[:, :, self._scopes[1, :], :, :].unsqueeze(3)

        # left + right with broadcasting: [n, d/2, c, 1, r] + [n, d/2, 1, c, r] -> [n, d/2, c, c, r]
        result = left + right

        # Put the two channel dimensions from the outer sum into one single dimension:
        # [n, d/2, c, c, r] -> [n, d/2, c * c, r]
        result = result.view(n, w, d_out, c * c, r)

        assert result.size() == (n, w, d_out, c * c, r)
        return result

    def sample(self, ctx: SamplingContext = None) -> Union[SamplingContext, th.Tensor]:
        raise NotImplementedError("sample() has been split up into sample_index_style() and sample_onehot_style()!"
                                  "Please choose one.")

    def sample_index_style(self, ctx: SamplingContext = None) -> SamplingContext:
        """Method to sample from this layer, based on the parents output.

        Args:
            ctx (SamplingContext):
                n: Number of samples.
                parent_indices (th.Tensor): Nodes selected by parent layer
                repetition_indices (th.Tensor): Repetitions selected by parent layer
        Returns:
            th.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """

        # If this is a root node
        if ctx.is_root:
            # Sampling starting at a CrossProduct layer means sampling each node in the layer.

            # There are oc * r * out_features nodes in this layer.
            # We sample across all repetitions at the same time, so the parent_indices tensor has a repetition dim.

            # The parent and repetition indices are also repeated by the number of samples requested
            # and by the number of conditionals the CSPN weights have been set to.
            # In the RatSpn case, the number of conditionals (abbreviated by w) is 1.
            indices = self.unraveled_channel_indices.data.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
            # indices is [nr_nodes=oc, 1, 1, cardinality, 1]
            indices = indices.repeat(
                1, ctx.n, self.num_conditionals, self.in_features//self.cardinality, self.num_repetitions
            )
            # indices is [nr_nodes=oc, n, w, in_features, r]
            oc, _ = self.unraveled_channel_indices.shape

            # repetition indices are left empty because they are implicitly given in parent_indices
        else:
            nr_nodes, n, w, d = ctx.parent_indices.shape[:4]
            # Map flattened indexes back to coordinates to obtain the chosen input_channel for each feature
            indices = self.unraveled_channel_indices[ctx.parent_indices]
            if ctx.parent_indices.dim() == 5:
                r = ctx.parent_indices.size(4)
                indices = indices.permute(0, 1, 2, 3, 5, 4).reshape(nr_nodes, n, w, d * self.cardinality, r)
            else:
                indices = indices.view(nr_nodes, n, w, -1)

        # Remove padding
        if self._pad:
            indices = indices[:, :, :, : -self._pad]

        ctx.parent_indices = indices
        return ctx

    def sample_onehot_style(self, ctx: SamplingContext = None) -> SamplingContext:
        """Method to sample from this layer, based on the parents output.

        Args:
            ctx (SamplingContext):
                n: Number of samples.
                parent_indices (th.Tensor): Nodes selected by parent layer
                repetition_indices (th.Tensor): Repetitions selected by parent layer
        Returns:
            th.Tensor: Index into tensor which paths should be followed.
                          Output should be of size: in_features, out_channels.
        """

        # If this is a root node
        if ctx.is_root:
            # Sampling starting at a CrossProduct layer means sampling each node in the layer.

            # There are oc * r * out_features nodes in this layer.
            # We sample across all repetitions at the same time, so the parent_indices tensor has a repetition dim.

            # The parent and repetition indices are also repeated by the number of samples requested
            # and by the number of conditionals the CSPN weights have been set to.
            # In the RatSpn case, the number of conditionals (abbreviated by w) is 1.
            indices = self.one_hot_in_channel_mapping.data.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
            # indices is [nr_nodes=oc, 1, 1, cardinality, 1]
            indices = indices.repeat(
                1, ctx.n, self.num_conditionals, self.in_features//self.cardinality, 1, self.num_repetitions
            )
            # indices is [nr_nodes=oc, n, w, in_features, r]
        else:
            nr_nodes, n, w, d, oc, r = ctx.parent_indices.shape
            indices = ctx.parent_indices.permute(0, 1, 2, 3, 5, 4).unsqueeze(-1).unsqueeze(-1)
            indices = indices * self.one_hot_in_channel_mapping
            # Shape [nr_nodes, n, w, d, r, oc, 2, ic]
            indices = indices.sum(dim=5)
            # Shape [nr_nodes, n, w, d, r, 2, ic]
            indices = indices.permute(0, 1, 2, 3, 5, 6, 4)  # [nr_nodes, n, w, d, 2, ic, r]
            indices = indices.reshape(nr_nodes, n, w, d * self.cardinality, self.in_channels, r)

        # Remove padding
        if self._pad:
            indices = indices[:, :, :, : -self._pad]

        ctx.parent_indices = indices
        return ctx

    def consolidate_weights(self, parent_weights):
        """
            This function takes the weights from the parent sum nodes meant for this product layer and recalculates
            them as if they would directly weigh the child sum nodes of this product layer.
            This turns the sum-prod-sum chain into a hierarchical mixture: sum-sum.
        """
        n, d, p_ic, p_oc, r = parent_weights.shape
        assert p_ic == self.in_channels**2, \
            "Number of parent input channels isn't the squared input channels of this product layer!"
        assert d*2 == self.in_features, \
            "Number of input features isn't double the output features of this product layer!"
        parent_weights = parent_weights.softmax(dim=2)
        parent_weights = parent_weights.view(n, d, self.in_channels, self.in_channels, p_oc, r)
        left_sums_weights = parent_weights.sum(dim=3)
        right_sums_weights = parent_weights.sum(dim=2)
        # left_sums_weights contains the consolidated weights of each parent's in_feature regarding the left sets of
        # sum nodes for that feature. right_sums_weights analogously for the right sets.
        # We can't simply concatenate them along the 1. dimension because this would mix up the order of in_features
        # of this layer. Along dim 1, we need left[0], right[0], left[1], right[1] => we need to interleave them
        parent_weights = th.stack((left_sums_weights, right_sums_weights), dim=2)
        parent_weights = parent_weights.view(n, self.in_features, self.in_channels, p_oc, r)
        return parent_weights

    def __repr__(self):
        return "CrossProduct(in_features={}, out_shape={})".format(self.in_features, self.out_shape)
