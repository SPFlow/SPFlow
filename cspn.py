import logging
from typing import Dict, Type

import numpy as np
import torch as th
import torch.nn.functional as F
from dataclasses import dataclass
from torch import nn

from layers import CrossProduct, Sum
from distributions import GaussianMixture

from rat_spn import RatSpn, RatSpnConfig
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def print_cspn_params(cspn):
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params in CSPN: {count_params(cspn)}")
    print(f"Params to extract features from the conditional: {count_params(cspn.feat_layers)}")
    print(f"Params in MLP for the sum params, excluding the heads: {count_params(cspn.sum_layers)}")
    print(f"Params in the heads of the sum param MLPs: {sum([count_params(head) for head in cspn.sum_param_heads])}")
    print(f"Params in MLP for the dist params, excluding the heads: {count_params(cspn.dist_layers)}")
    print(f"Params in the heads of the dist param MLPs: "
          f"{count_params(cspn.dist_mean_head) + count_params(cspn.dist_std_head)}")


@dataclass
class CspnConfig(RatSpnConfig):
    F_cond: tuple = 0
    feat_layers: list = None
    conv_kernel_size: int = 5
    conv_pooling_kernel_size: int = 3
    conv_pooling_stride: int = 3
    sum_param_layers: list = None
    dist_param_layers: list = None
    cond_layers_inner_act: Type[nn.Module] = nn.ReLU

    def __setattr__(self, key, value):
        if hasattr(self, key):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"CspnConfig object has no attribute {key}")


class CSPN(RatSpn):
    def __init__(self, config: CspnConfig):
        """
        Create a CSPN

        Args:
            config (CspnConfig): Cspn configuration object.
        """
        super().__init__(config=config)
        self.config: CspnConfig = config
        self.dist_std_head = None
        self.dist_mean_head = None
        self.dist_layers = None
        self.sum_param_heads = None
        self.sum_layers = None
        self.feat_layers = None
        self.replace_layer_params()
        self.create_feat_layers(config.F_cond)

    def forward(self, x: th.Tensor, condition: th.Tensor = None) -> th.Tensor:
        """
        Forward pass through RatSpn. Computes the conditional log-likelihood P(X | C).

        Args:
            x: Input of shape [batch, weight_sets, in_features, channel].
                batch: Number of samples per weight set (= per conditional in the CSPN sense).
                weight_sets: In CSPNs, weights are different for each conditional. In RatSpn, this is 1.

        Returns:
            th.Tensor: Conditional log-likelihood P(X | C) of the input.
        """
        if condition is not None:
            self.set_weights(condition)
        weight_sets = self._leaf.base_leaf.means.shape[0]
        if x.dim() == 3:
            assert x.shape[1] == weight_sets, \
                f"The input data shape says there are {x.shape[1]} samples of each conditional. " \
                f"But there are only samples of {x.shape[1]} conditionals, " \
                f"while the CSPN weights expect {weight_sets} conditionals! " \
                f"Did you forget to set the weights of the CSPN?"
        elif x.dim() == 2:
            assert x.shape[0] == weight_sets, \
                f"Dim of input is 2. This means that each input sample belongs to one conditional. " \
                f"But the number of samples ({x.shape[0]}) doesn't match the number of conditionals ({weight_sets})!"
            x = x.unsqueeze(0)

        return super().forward(x)

    def vi_entropy_approx(self, condition: th.Tensor = None, **kwargs) -> th.Tensor:
        if condition is not None:
            self.set_weights(condition)
        return super().vi_entropy_approx(**kwargs)

    def consolidate_weights(self, condition=None):
        if condition is not None:
            self.set_weights(condition)
        return super().consolidate_weights()

    def compute_moments(self, condition=None):
        if condition is not None:
            self.set_weights(condition)
        return super().compute_moments()

    def compute_gradients(self, x, with_log_prob_x=False, condition=None):
        if condition is not None:
            self.set_weights(condition)
        return super().compute_gradients(x, with_log_prob_x)

    def entropy_taylor_approx(self, condition=None, components=3):
        """
            Calculates the Taylor series approximation of the entropy up to a given order.
            The first order of the approximation is zero.
            'components' is the number of Taylor series terms that aren't zero, starting at the zero'th order.
        """
        # assert isinstance(self._inner_layers[0], Sum), "First layer after the leaf layer must be a sum layer!"
        if condition is not None:
            self.set_weights(condition)
        self.consolidate_weights(condition=None)
        moments = super().compute_moments(order=components)
        mean = moments[0]
        # Gradients are all evaluated at the mean of the SPN
        # grad, ggrad, gggrad, log_p_mean = super().compute_gradients(mean, with_log_prob_x=True, order=components)
        grads = super().compute_gradients(mean, with_log_prob_x=True, order=components)
        log_p_mean = grads[-1]
        entropy = grad = inv_sq_mean_prob = ggrad = inv_mean_prob = 0  # To satisfy the IDE
        if components >= 1:
            H_0 = - log_p_mean
            entropy = H_0
        if components >= 2:
            var = moments[1]
            grad, ggrad = grads[0:2]
            inv_mean_prob = (-log_p_mean).exp()
            inv_sq_mean_prob = (-2 * log_p_mean).exp()
            ggrad_log = -inv_sq_mean_prob * grad + inv_mean_prob * ggrad
            H_2 = - (ggrad_log * var) / 2
            entropy += H_2
        if components >= 3:
            skew = moments[2]
            gggrad = grads[2]
            inv_cub_mean_prob = (-3 * log_p_mean).exp()
            gggrad_log = 2 * inv_cub_mean_prob * grad - 2 * inv_sq_mean_prob * ggrad + inv_mean_prob * gggrad
            H_3 = - (gggrad_log * skew) / 6
            entropy += H_3

        # grad_log = inv_mean_prob * grad
        entropy = entropy.sum(dim=1)
        return entropy

    def iterative_gmm_entropy_lb(self, condition=None, reduction='mean'):
        """
            Calculate the entropy lower bound of the first-level mixtures.
            See "On Entropy Approximation for Gaussian Mixture Random Vectors" Huber et al. 2008, Theorem 2
        """
        if condition is not None:
            self.set_weights(condition)
        return self._leaf.iterative_gmm_entropy_lb(reduction)

    def gmm_entropy_lb(self, condition=None, reduction='mean'):
        """
            Calculate the entropy lower bound of the first-level mixtures.
            See "On Entropy Approximation for Gaussian Mixture Random Vectors" Huber et al. 2008, Theorem 2
        """
        if condition is not None:
            self.set_weights(condition)
        return self._leaf.gmm_entropy_lb(reduction)

    def leaf_entropy_taylor_approx(self, condition=None, components=3):
        """
            Calculate the entropy lower bound of the first-level mixtures.
            See "On Entropy Approximation for Gaussian Mixture Random Vectors" Huber et al. 2008, Theorem 2
        """
        if condition is not None:
            self.set_weights(condition)
        return self._leaf.entropy_taylor_approx(components=components)

    def sum_node_entropies(self, condition=None, reduction='mean'):
        """
            Calculate the entropies of the hidden categorical random variables in the sum nodes
        """
        if condition is not None:
            self.set_weights(condition)
        return super().sum_node_entropies(reduction)

    def sample(self, mode: str = None, condition: th.Tensor = None, class_index=None,
               evidence: th.Tensor = None, **kwargs):
        """
        Sample from the random variable encoded by the CSPN.

        Args:
            mode: Two sampling modes are supported:
                'index': Sampling mechanism with indexes, which are non-differentiable.
                'onehot': This sampling mechanism work with one-hot vectors, grouped into tensors.
                          This way of sampling is differentiable, but also takes almost twice as long.
            condition (th.Tensor): Batch of conditionals.
            class_index: See doc of RatSpn.sample_index_style()
            evidence: See doc of RatSpn.sample_index_style()
        """
        if condition is not None:
            self.set_weights(condition)
        assert class_index is None or condition.shape[0] == len(class_index), \
            "The batch size of the condition must equal the length of the class index list if they are provided!"
        # TODO add assert to check dimension of evidence, if given.

        return super().sample(mode=mode, class_index=class_index, evidence=evidence, **kwargs)

    def sample_index_style(self, **kwargs):
        return self.sample(mode='index', **kwargs)

    def sample_onehot_style(self, **kwargs):
        return self.sample(mode='onehot', **kwargs)

    def replace_layer_params(self):
        for layer in self._inner_layers:
            if isinstance(layer, Sum):
                placeholder = th.zeros_like(layer.weights)
                del layer.weights
                layer.weights = placeholder
        placeholder = th.zeros_like(self.root.weights)
        del self.root.weights
        self.root.weights = placeholder

        placeholder = th.zeros_like(self._sampling_root.weights)
        del self._sampling_root.weights
        self._sampling_root.weights = placeholder

        if isinstance(self._leaf, GaussianMixture):
            placeholder = th.zeros_like(self._leaf.sum.weights)
            del self._leaf.sum.weights
            self._leaf.sum.weights = placeholder

        placeholder = th.zeros_like(self._leaf.base_leaf.means)
        del self._leaf.base_leaf.means
        del self._leaf.base_leaf.stds
        self._leaf.base_leaf.means = placeholder
        self._leaf.base_leaf.stds = placeholder

    def create_feat_layers(self, feature_dim: tuple):
        assert len(feature_dim) == 3 or len(feature_dim) == 1, \
            f"Don't know how to construct feature extraction layers for features of dim {len(feature_dim)}."
        if len(feature_dim) == 3:
            # feature_dim = (channels, rows, columns)
            assert False, "Adapt conv layer feat extraction to config.feat_layer_sizes"
            conv_kernel = self.config.conv_kernel_size
            pool_kernel = self.config.conv_pooling_kernel_size
            pool_stride = self.config.conv_pooling_stride
            nr_feat_layers = 0
            conv_layers = [] if nr_feat_layers > 0 else [nn.Identity()]
            for j in range(nr_feat_layers):
                # feature_dim = [int(np.floor((n - (pool_kernel-1) - 1)/pool_stride + 1)) for n in feature_dim]
                in_channels = feature_dim[0]
                if j == nr_feat_layers-1:
                    out_channels = 1
                else:
                    out_channels = feature_dim[0]
                conv_layers += [nn.Conv2d(in_channels, out_channels,
                                          kernel_size=(conv_kernel, conv_kernel), padding='same'),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
                                nn.Dropout()]
            self.feat_layers = nn.Sequential(*conv_layers)
        elif len(feature_dim) == 1:
            if self.config.feat_layers:
                feat_layers = []
                layer_sizes = [feature_dim[0]] + self.config.feat_layers
                for j in range(len(layer_sizes) - 1):
                    feat_layers += [nn.Linear(layer_sizes[j], layer_sizes[j+1]),
                                    self.config.cond_layers_inner_act()]
            else:
                feat_layers = [nn.Identity()]
            self.feat_layers = nn.Sequential(*feat_layers)

        output_activation = nn.Identity

        feature_dim = int(np.prod(self.feat_layers(th.ones((1, *feature_dim))).shape))
        print(f"The feature extraction layer for the CSPN conditional reduce the {int(np.prod(feature_dim))} "
              f"inputs (e.g. pixels in an image) down to {feature_dim} features. These are the inputs of the "
              f"MLPs which set the sum and dist params.")
        # sum_layer_sizes = [int(feature_dim * 10 ** (-i)) for i in range(1 + self.config.fc_sum_param_layers)]
        sum_layer_sizes = [feature_dim]
        if self.config.sum_param_layers:
            sum_layers = []
            sum_layer_sizes += self.config.sum_param_layers
            for j in range(len(sum_layer_sizes) - 1):
                act = self.config.cond_layers_inner_act if j < len(sum_layer_sizes) - 2 else output_activation
                sum_layers += [nn.Linear(sum_layer_sizes[j], sum_layer_sizes[j + 1]), act()]
        else:
            sum_layers = [nn.Identity()]
        self.sum_layers = nn.Sequential(*sum_layers)

        self.sum_param_heads = nn.ModuleList()
        for layer in self._inner_layers:
            if isinstance(layer, Sum):
                self.sum_param_heads.append(nn.Linear(sum_layer_sizes[-1], layer.weights.numel()))
                print(f"Sum layer has {layer.weights.numel()} weights.")
        self.sum_param_heads.append(nn.Linear(sum_layer_sizes[-1], self.root.weights.numel()))
        print(f"Root sum layer has {self.root.weights.numel()} weights.")

        if isinstance(self._leaf, GaussianMixture):
            self.sum_param_heads.append(nn.Linear(sum_layer_sizes[-1], self._leaf.sum.weights.numel()))
            print(f"A param head was added for the sum layer of the GaussianMixture leaves, "
                  f"having {self._leaf.sum.weights.numel()} weights.")

        # dist_layer_sizes = [int(feature_dim * 10 ** (-i)) for i in range(1 + self.config.fc_dist_param_layers)]
        dist_layer_sizes = [feature_dim]
        if self.config.dist_param_layers:
            dist_layers = []
            dist_layer_sizes += self.config.dist_param_layers
            for j in range(len(dist_layer_sizes) - 1):
                act = self.config.cond_layers_inner_act if j < len(dist_layer_sizes) - 2 else output_activation
                dist_layers += [nn.Linear(dist_layer_sizes[j], dist_layer_sizes[j + 1]), act()]
        else:
            dist_layers = [nn.Identity()]
        self.dist_layers = nn.Sequential(*dist_layers)

        self.dist_mean_head = nn.Linear(dist_layer_sizes[-1], self._leaf.base_leaf.means.numel())
        self.dist_std_head = nn.Linear(dist_layer_sizes[-1], self._leaf.base_leaf.stds.numel())
        print(f"Dist layer has {self._leaf.base_leaf.means.numel()} + {self._leaf.base_leaf.stds.numel()} weights.")

    def create_one_hot_in_channel_mapping(self):
        for lay in self._inner_layers:
            if isinstance(lay, CrossProduct):
                lay.one_hot_in_channel_mapping = F.one_hot(lay.unraveled_channel_indices).float().requires_grad_(False)

    def set_no_tanh_log_prob_correction(self):
        self._leaf.base_leaf.set_no_tanh_log_prob_correction()

    def set_weights(self, feat_inp: th.Tensor):
        """
            Sets the weights of the sum and dist nodes, using the input from the conditional passed through the
            feature extraction layers.
            The weights of the sum nodes are normalized in log space (log-softmaxed) over the input channel dimension.
            The distribution parameters are bounded as well via the bounding function of the leaf layer.
            So in the RatSpn class, any normalizing and bounding must only be done if the weights are of dimension 4,
            meaning that it is not a Cspn.
        """
        num_conditionals = feat_inp.shape[0]
        features = self.feat_layers(feat_inp)
        features = features.flatten(start_dim=1)
        sum_weights_pre_output = self.sum_layers(features)

        LOG_STD_MAX = 2
        LOG_STD_MIN = -20

        # Set normalized sum node weights of the inner RatSpn layers
        i = 0
        for layer in self._inner_layers:
            if isinstance(layer, Sum):
                weight_shape = (num_conditionals, layer.in_features, layer.in_channels, layer.out_channels, layer.num_repetitions)
                weights = self.sum_param_heads[i](sum_weights_pre_output).view(weight_shape)
                weights = th.clamp(weights, LOG_STD_MIN, LOG_STD_MAX)
                layer.weights = F.log_softmax(weights, dim=2)
                i += 1
            else:
                layer.num_conditionals = num_conditionals

        # Set normalized weights of the root sum layer
        weight_shape = (num_conditionals, self.root.in_features, self.root.in_channels, self.root.out_channels, self.root.num_repetitions)
        weights = self.sum_param_heads[i](sum_weights_pre_output).view(weight_shape)
        weights = th.clamp(weights, LOG_STD_MIN, LOG_STD_MAX)
        self.root.weights = F.log_softmax(weights, dim=2)

        # Sampling root weights need to have 5 dims as well
        weight_shape = (num_conditionals, 1, 1, 1, 1)
        self._sampling_root.weights = th.ones(weight_shape).to(self.root.weights.device)
        self._sampling_root.weights = self._sampling_root.weights.mul_(1/self.config.C).log_()

        # Set normalized weights of the Gaussian Mixture leaf layer if it exists.
        if isinstance(self._leaf, GaussianMixture):
            self._leaf.reset_moment_cache()
            weight_shape = (num_conditionals, self._leaf.sum.in_features, self._leaf.sum.in_channels,
                            self._leaf.sum.out_channels, self._leaf.sum.num_repetitions)
            weights = self.sum_param_heads[i+1](sum_weights_pre_output).view(weight_shape)
            weights = th.clamp(weights, LOG_STD_MIN, LOG_STD_MAX)
            self._leaf.sum.weights = F.log_softmax(weights, dim=2)

        # Set bounded weights of the Gaussian distributions in the leaves
        dist_param_shape = (num_conditionals, self._leaf.base_leaf.in_features, self.config.I, self.config.R)
        dist_weights_pre_output = self.dist_layers(features)
        dist_means = self.dist_mean_head(dist_weights_pre_output).view(dist_param_shape)
        dist_stds = self.dist_std_head(dist_weights_pre_output).view(dist_param_shape)
        self._leaf.base_leaf.means = dist_means
        self._leaf.base_leaf.stds = dist_stds
        self._leaf.base_leaf.set_bounded_dist_params()
