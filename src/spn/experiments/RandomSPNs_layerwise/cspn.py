import logging
from typing import Dict, Type

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch import nn

from spn.algorithms.layerwise.distributions import Leaf
from spn.algorithms.layerwise.layers import CrossProduct, Sum
from spn.algorithms.layerwise.type_checks import check_valid
from spn.algorithms.layerwise.utils import provide_evidence, SamplingContext
from spn.experiments.RandomSPNs_layerwise.distributions import IndependentMultivariate, RatNormal, truncated_normal_

from rat_spn import RatSpn, RatSpnConfig
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class CspnConfig(RatSpnConfig):
    F_cond: tuple = 0
    nr_feat_layers: int = 1
    conv_kernel_size: int = 5
    conv_pooling_kernel_size: int = 3
    conv_pooling_stride: int = 3
    fc_sum_param_layers: int = 1
    fc_dist_param_layers: int = 1

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
        config.first_layer_sum = True  # This must be True so we can calculate entropy
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

    def replace_layer_params(self):
        for layer in self._inner_layers:
            if isinstance(layer, Sum):
                placeholder = torch.zeros_like(layer.weights)
                del layer.weights
                layer.weights = placeholder
        placeholder = torch.zeros_like(self.root.weights)
        del self.root.weights
        self.root.weights = placeholder

        placeholder = torch.zeros_like(self._leaf.base_leaf.means)
        del self._leaf.base_leaf.means
        del self._leaf.base_leaf.stds
        self._leaf.base_leaf.means = placeholder
        self._leaf.base_leaf.stds = placeholder

    def create_feat_layers(self, feature_input_dim: tuple):
        nr_feat_layers = self.config.nr_feat_layers
        conv_kernel = self.config.conv_kernel_size
        pool_kernel = self.config.conv_pooling_kernel_size
        pool_stride = self.config.conv_pooling_stride
        feature_dim = feature_input_dim
        assert len(feature_dim) == 3 or len(feature_dim) == 1, \
            f"Don't know how to construct feature extraction layers for {len(feature_dim)} features."
        if len(feature_dim) == 3:
            # feature_dim = (channels, rows, columns)
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
            feat_layers = [] if nr_feat_layers > 0 else [nn.Identity()]
            for j in range(nr_feat_layers):
                feat_layers += [nn.Linear(feature_dim[0], feature_dim[0]), nn.ReLU()]
            self.feat_layers = nn.Sequential(*feat_layers)

        activation = nn.ReLU
        output_activation = nn.Identity

        feature_dim = int(np.prod(self.feat_layers(torch.ones((1, *feature_input_dim))).shape))
        print(f"The feature extraction layer for the CSPN conditional reduce the {int(np.prod(feature_input_dim))} "
              f"inputs (e.g. pixels in an image) down to {feature_dim} features. These are the inputs of the "
              f"MLPs which set the sum and dist params.")
        # sum_layer_sizes = [int(feature_dim * 10 ** (-i)) for i in range(1 + self.config.fc_sum_param_layers)]
        sum_layer_sizes = [feature_dim for _ in range(1 + self.config.fc_sum_param_layers)]
        sum_layers = []
        for j in range(len(sum_layer_sizes) - 1):
            act = activation if j < len(sum_layer_sizes) - 2 else output_activation
            sum_layers += [nn.Linear(sum_layer_sizes[j], sum_layer_sizes[j + 1]), act()]
        self.sum_layers = nn.Sequential(*sum_layers)

        self.sum_param_heads = nn.ModuleList()
        for layer in self._inner_layers:
            if isinstance(layer, Sum):
                self.sum_param_heads.append(nn.Linear(sum_layer_sizes[-1], layer.weights.numel()))
                print(f"Sum layer has {layer.weights.numel()} weights.")
        self.sum_param_heads.append(nn.Linear(sum_layer_sizes[-1], self.root.weights.numel()))
        print(f"Root sum layer has {self.root.weights.numel()} weights.")

        # dist_layer_sizes = [int(feature_dim * 10 ** (-i)) for i in range(1 + self.config.fc_dist_param_layers)]
        dist_layer_sizes = [feature_dim for _ in range(1 + self.config.fc_dist_param_layers)]
        dist_layers = []
        for j in range(len(dist_layer_sizes) - 1):
            act = activation if j < len(dist_layer_sizes) - 2 else output_activation
            dist_layers += [nn.Linear(dist_layer_sizes[j], dist_layer_sizes[j + 1]), act()]
        self.dist_layers = nn.Sequential(*dist_layers)

        self.dist_mean_head = nn.Linear(dist_layer_sizes[-1], self._leaf.base_leaf.means.numel())
        self.dist_std_head = nn.Linear(dist_layer_sizes[-1], self._leaf.base_leaf.stds.numel())
        print(f"Dist layer has {self._leaf.base_leaf.means.numel()} + {self._leaf.base_leaf.stds.numel()} weights.")

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        if condition is not None:
            self.set_weights(condition)
        return super().forward(x)

    def gmm_entropy_lb(self, condition=None, reduction='mean'):
        """
            Calculate the entropy lower bound of the first-level mixtures.
            See "On Entropy Approximation for Gaussian Mixture Random Vectors" Huber et al. 2008, Theorem 2
        """
        assert isinstance(self._inner_layers[0], Sum), "First layer after the leaf layer must be a sum layer!"
        if condition is not None:
            self.set_weights(condition)
        return super().gmm_entropy_lb(reduction)

    def sum_node_entropies(self, condition=None, reduction='mean'):
        """
            Calculate the entropies of the hidden categorical random variables in the sum nodes
        """
        if condition is not None:
            self.set_weights(condition)
        return super().sum_node_entropies(reduction)

    def sample(self, condition: torch.Tensor = None, class_index=None,
               evidence: torch.Tensor = None, is_mpe: bool = False, **kwargs):
        """
        Sample from the random variable encoded by the CSPN.

        Args:
            condition (torch.Tensor): Batch of conditionals.
        """
        if condition is not None:
            self.set_weights(condition)
        assert class_index is None or condition.shape[0] == len(class_index), \
            "The batch size of the condition must equal the length of the class index list if they are provided!"
        # TODO add assert to check dimension of evidence, if given.

        batch_size = self.root.weights.shape[0]
        return super().sample(batch_size, class_index, evidence, is_mpe, **kwargs)

    def squared_weights(self, reduction='mean'):
        inner_sum_weight_decay_losses = []
        for layer in self._inner_layers:
            if isinstance(layer, Sum):
                # weights [N x D x IC x OC x R]
                squared_weights = layer.weights ** 2
                if reduction == 'mean':
                    squared_weights = squared_weights.mean()
                elif reduction == 'sum':
                    squared_weights = squared_weights.sum()
                inner_sum_weight_decay_losses.append(squared_weights)
        root_sum_weight_decay_loss = self.root.weights ** 2
        if reduction == 'mean':
            root_sum_weight_decay_loss = root_sum_weight_decay_loss.mean()
        elif reduction == 'sum':
            root_sum_weight_decay_loss = root_sum_weight_decay_loss.sum()
        return inner_sum_weight_decay_losses, root_sum_weight_decay_loss

    def set_weights(self, feat_inp):
        batch_size = feat_inp.shape[0]
        features = self.feat_layers(feat_inp)
        features = features.flatten(start_dim=1)
        sum_weights_pre_output = self.sum_layers(features)

        i = 0
        for layer in self._inner_layers:
            if isinstance(layer, Sum):
                weight_shape = (batch_size, layer.in_features, layer.in_channels, layer.out_channels, layer.num_repetitions)
                weights = self.sum_param_heads[i](sum_weights_pre_output).view(weight_shape)
                layer.weights = weights
                i += 1
        weight_shape = (batch_size, self.root.in_features, self.root.in_channels, self.root.out_channels, self.root.num_repetitions)
        weights = self.sum_param_heads[i](sum_weights_pre_output).view(weight_shape)
        self.root.weights = weights

        dist_param_shape = (batch_size, self._leaf.base_leaf.in_features, self.config.I, self.config.R)
        dist_weights_pre_output = self.dist_layers(features)
        dist_means = self.dist_mean_head(dist_weights_pre_output).view(dist_param_shape)
        dist_stds = self.dist_std_head(dist_weights_pre_output).view(dist_param_shape)
        self._leaf.base_leaf.means = dist_means
        self._leaf.base_leaf.stds = dist_stds
