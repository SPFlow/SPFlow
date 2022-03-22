import logging
from typing import Dict, Tuple, List, Optional

import math
import numpy as np
import torch as th
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from base_distributions import Leaf, dist_forward
from layers import Product, Sum
from type_checks import check_valid
from utils import SamplingContext

logger = logging.getLogger(__name__)


class RatNormal(Leaf):
    """ Implementation as in RAT-SPN

    Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(
        self,
        in_features: int,
        out_channels: int,
        num_repetitions: int = 1,
        dropout: float = 0.0,
        tanh_squash: bool = False,
        min_sigma: float = None,
        max_sigma: float = None,
        min_mean: float = None,
        max_mean: float = None,
        no_tanh_log_prob_correction: bool = False,
    ):
        """Create a gaussian layer.

        Args:
            in_features: Number of input features.
            out_channels: Number of parallel representations for each input feature.
            tanh_bounds: If set, a correction term is applied to the log probs.
        """
        super().__init__(in_features, out_channels, num_repetitions, dropout)

        # Create gaussian means and stds
        self.means = nn.Parameter(th.randn(1, in_features, out_channels, num_repetitions))

        self._tanh_squash = tanh_squash
        self._no_tanh_log_prob_correction = no_tanh_log_prob_correction

        if min_sigma is not None and max_sigma is not None:
            # Init from normal
            self.stds = nn.Parameter(th.randn(1, in_features, out_channels, num_repetitions))
        else:
            # Init uniform between 0 and 1
            self.stds = nn.Parameter(th.rand(1, in_features, out_channels, num_repetitions))

        self.min_sigma = check_valid(min_sigma, float, 0.0, max_sigma, allow_none=True)
        self.max_sigma = check_valid(max_sigma, float, min_sigma, allow_none=True)
        self.min_mean = check_valid(min_mean, float, upper_bound=max_mean, allow_none=True)
        self.max_mean = check_valid(max_mean, float, min_mean, allow_none=True)

        self._dist_params_are_bounded = False

    def set_no_tanh_log_prob_correction(self):
        self._no_tanh_log_prob_correction = False

    def forward(self, x):
        if x.dim() == 4:
            # Create extra output-channel dimension
            x = x.unsqueeze(3)

        correction = None
        if self._tanh_squash and not self._no_tanh_log_prob_correction:
            # This correction term assumes that the input is from a distribution with infinite support
            correction = 2 * (np.log(2) - x - F.softplus(-2 * x))
            # This correction term assumes the input to be squashed already
            # correction = th.log(1 - x**2 + 1e-6)

        d = self._get_base_distribution()
        x = d.log_prob(x)  # Shape: [n, w, d, oc, r]

        if self._tanh_squash and not self._no_tanh_log_prob_correction:
            x -= correction

        x = self._marginalize_input(x)
        x = self._apply_dropout(x)

        return x

    def sample(self, ctx: SamplingContext = None):
        raise NotImplementedError("sample() has been split up into sample_index_style() and sample_onehot_style()!"
                                  "Please choose one.")

    def sample_index_style(self, ctx: SamplingContext = None) -> th.Tensor:
        """
        Perform sampling, given indices from the parent layer that indicate which of the multiple representations
        for each input shall be used.
        """
        if ctx.is_root:
            if ctx.is_mpe:
                samples: th.Tensor = self.means.unsqueeze(0).expand(ctx.n, -1, -1, -1, -1)
            else:
                gauss = dist.Normal(self.means, self.stds)
                samples: th.Tensor = gauss.rsample(sample_shape=(ctx.n,))
        else:
            nr_nodes, n, w = ctx.parent_indices.shape[:3]
            _, d, i, r = self.means.shape
            selected_means = self.means.unsqueeze(0).unsqueeze(0).expand(nr_nodes, n, -1, -1, -1, -1)
            selected_stds = self.stds.unsqueeze(0).unsqueeze(0).expand(nr_nodes, n, -1, -1, -1, -1)

            if ctx.repetition_indices is not None:
                rep_ind = ctx.repetition_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                rep_ind = rep_ind.expand(-1, -1, -1, d, i, -1)
                selected_means = th.gather(selected_means, dim=-1, index=rep_ind).squeeze(-1)
                selected_stds = th.gather(selected_stds, dim=-1, index=rep_ind).squeeze(-1)

            # Select means and std in the output_channel dimension
            par_ind = ctx.parent_indices.unsqueeze(4)
            selected_means = th.gather(selected_means, dim=4, index=par_ind).squeeze(4)
            selected_stds = th.gather(selected_stds, dim=4, index=par_ind).squeeze(4)

            if ctx.is_mpe:
                samples = selected_means
            else:
                gauss = dist.Normal(selected_means, selected_stds)
                samples = gauss.rsample()

        return samples

    def sample_onehot_style(self, ctx: SamplingContext = None) -> th.Tensor:
        """
        Perform sampling, given indices from the parent layer that indicate which of the multiple representations
        for each input shall be used.
        """
        if ctx.is_root:
            if ctx.is_mpe:
                samples: th.Tensor = self.means.unsqueeze(0).expand(ctx.n, -1, -1, -1, -1)
            else:
                gauss = dist.Normal(self.means, self.stds)
                samples: th.Tensor = gauss.rsample(sample_shape=(ctx.n,))
        else:
            # ctx.parent_indices shape [nr_nodes, n, w, f, oc, r]
            # self.means shape [w, f, oc, r]
            selected_means = self.means * ctx.parent_indices
            assert ctx.parent_indices.detach().sum(4).max().item() == 1.0
            selected_means = selected_means.sum(4)
            if ctx.parent_indices.detach()[0, 0, 0, 0, :, :].sum().item() == 1.0:
                # Only one repetition is selected, remove repetition dim of parameters
                selected_means = selected_means.sum(-1)

            if ctx.is_mpe:
                samples = selected_means
            else:
                selected_stds = self.stds * ctx.parent_indices
                selected_stds = selected_stds.sum(4)
                if ctx.parent_indices.detach()[0, 0, 0, 0, :, :].sum().item() == 1.0:
                    # Only one repetition is selected, remove repetition dim of parameters
                    selected_stds = selected_stds.sum(-1)

                gauss = dist.Normal(selected_means, selected_stds)
                samples = gauss.rsample()

        return samples

    def set_bounded_dist_params(self):
        """
            Set the dist params to their bounded values. This is called by the Cspn.set_weights() function to
            save calling self.bounded_dist_params() later on.
        """
        self._dist_params_are_bounded = False
        self.means, self.stds = self.bounded_dist_params()
        self._dist_params_are_bounded = True

    def bounded_dist_params(self) -> Tuple[th.Tensor, th.Tensor]:
        if not self._dist_params_are_bounded:
            if self.min_sigma:
                if self.min_sigma < self.max_sigma:
                    sigma_ratio = th.sigmoid(self.stds)
                    sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma_ratio
                else:
                    sigma = 1.0
            else:
                LOG_STD_MAX = 2
                LOG_STD_MIN = -20
                sigma = th.clamp(self.stds, LOG_STD_MIN, LOG_STD_MAX).exp()

            means = self.means
            if self.max_mean:
                assert self.min_mean is not None
                # mean_range = self.max_mean - self.min_mean
                # means = th.sigmoid(self.means) * mean_range + self.min_mean
                means = th.clamp(means, self.min_mean, self.max_mean)

            return means, sigma
        else:
            return self.means, self.stds

    def _get_base_distribution(self) -> th.distributions.Distribution:
        means, sigma = self.bounded_dist_params()
        gauss = dist.Normal(means, sigma)
        return gauss

    def moments(self):
        """Get the mean and variance"""
        means, sigma = self.bounded_dist_params()
        return means, sigma.pow(2)

    def gradient(self, x: th.Tensor, order: int):
        """Get the gradient up to a given order at the point x"""
        assert order <= 3, "Gradient only implemented up to the third order!"

        if x.dim() == 3:  # Number of repetition dimension already exists
            x = x.unsqueeze(2)  # Shape [n, d, 1, r]
        elif x.dim() == 2:
            x = x.unsqueeze(2).unsqueeze(3)  # Shape: [n, d, 1, 1]
        d = self._get_base_distribution()
        prob = d.log_prob(x).exp_()
        inv_var = 1/d.variance
        a = inv_var * (x - d.mean)
        grads = []
        if order >= 1:
            grads += [- prob * a]
        if order >= 2:
            grads += [prob * (a**2 - inv_var)]
        if order >= 3:
            grads += [prob * (-a**3 + 3 * a * inv_var)]
        return grads


class IndependentMultivariate(Leaf):
    def __init__(
        self,
        in_features: int,
        out_channels: int,
        cardinality: int,
        num_repetitions: int = 1,
        dropout: float = 0.0,
        tanh_squash: bool = False,
        leaf_base_class: Leaf = RatNormal,
        leaf_base_kwargs: Dict = None,
    ):
        """
        Create multivariate distribution that only has non zero values in the covariance matrix on the diagonal.

        Args:
            out_channels: Number of parallel representations for each input feature.
            cardinality: Number of variables per gauss.
            in_features: Number of input features.
            dropout: Dropout probabilities.
            leaf_base_class (Leaf): The encapsulating base leaf layer class.

        """
        super(IndependentMultivariate, self).__init__(in_features, out_channels, num_repetitions, dropout)
        if leaf_base_kwargs is None:
            leaf_base_kwargs = {}

        self.base_leaf = leaf_base_class(
            out_channels=out_channels,
            in_features=in_features,
            dropout=dropout,
            num_repetitions=num_repetitions,
            tanh_squash=tanh_squash,
            **leaf_base_kwargs,
        )
        self._pad = (cardinality - self.in_features % cardinality) % cardinality
        # Number of input features for the product needs to be extended depending on the padding applied here
        prod_in_features = in_features + self._pad
        self.prod = Product(
            in_features=prod_in_features, cardinality=cardinality, num_repetitions=num_repetitions
        )

        self.cardinality = check_valid(cardinality, int, 1, in_features + 1)
        self.out_shape = f"(N, {self.prod._out_features}, {out_channels}, {self.num_repetitions})"

    @property
    def out_features(self):
        return self.prod._out_features

    def _init_weights(self):
        if isinstance(self.base_leaf, RatNormal):
            truncated_normal_(self.base_leaf.stds, std=0.5)

    def pad_input(self, x: th.Tensor):
        if self._pad:
            x = F.pad(x, pad=[0, 0, 0, 0, 0, self._pad], mode="constant", value=0.0)
        return x

    @property
    def pad(self):
        return self._pad

    def forward(self, x: th.Tensor, reduction='sum'):
        # Pass through base leaf
        x = self.base_leaf(x)
        x = self.pad_input(x)

        # Pass through product layer
        x = self.prod(x, reduction=reduction)
        return x

    def entropy(self):
        ent = self.base_leaf.entropy().unsqueeze(0)
        ent = self.pad_input(ent)
        ent = self.prod(ent, reduction='sum')
        return ent

    def _get_base_distribution(self):
        raise Exception("IndependentMultivariate does not have an explicit PyTorch base distribution.")

    def sample(self, ctx: SamplingContext = None):
        raise NotImplementedError("sample() has been split up into sample_index_style() and sample_onehot_style()!"
                                  "Please choose one.")

    def sample_index_style(self, ctx: SamplingContext = None) -> th.Tensor:
        if not ctx.is_root:
            ctx = self.prod.sample(ctx=ctx)

            # Remove padding
            if self._pad:
                ctx.parent_indices = ctx.parent_indices[:, :, :, :-self._pad]

        samples = self.base_leaf.sample_index_style(ctx=ctx)
        return samples

    def sample_onehot_style(self, ctx: SamplingContext = None) -> th.Tensor:
        if not ctx.is_root:
            ctx = self.prod.sample(ctx=ctx)

            # Remove padding
            if self._pad:
                ctx.parent_indices = ctx.parent_indices[:, :, :, :-self._pad]

        samples = self.base_leaf.sample_onehot_style(ctx=ctx)
        return samples

    def moments(self):
        """Get the mean, variance and third central moment (unnormalized skew)"""
        return [self.prod(self.pad_input(m), reduction=None) for m in self.base_leaf.moments()]

    def gradient(self, x: th.Tensor, order: int):
        """Get the gradient up to the given order at the point x"""
        return [self.prod(self.pad_input(g), reduction=None) for g in self.base_leaf.gradient(x, order)]

    def __repr__(self):
        return f"IndependentMultivariate(in_features={self.in_features}, out_channels={self.out_channels}, dropout={self.dropout}, cardinality={self.cardinality}, out_shape={self.out_shape})"


class GaussianMixture(IndependentMultivariate):
    def __init__(
            self,
            in_features: int,
            gmm_modes: int,
            out_channels: int,
            cardinality: int,
            num_repetitions: int = 1,
            dropout: float = 0.0,
            leaf_base_class: Leaf = RatNormal,
            leaf_base_kwargs: Dict = None,
    ):
        super(GaussianMixture, self).__init__(
            in_features=in_features, out_channels=gmm_modes, cardinality=cardinality, num_repetitions=num_repetitions,
            dropout=dropout, leaf_base_class=leaf_base_class, leaf_base_kwargs=leaf_base_kwargs)
        self.sum = Sum(in_features=self.out_features, in_channels=gmm_modes, num_repetitions=num_repetitions,
                       out_channels=out_channels, dropout=dropout)
        self._cached_moments = None

    def reset_moment_cache(self):
        self._cached_moments = None

    def forward(self, x: th.Tensor, reduction='sum'):
        x = super().forward(x=x, reduction=reduction)
        if reduction is None:
            x = self._weighted_sum(x)
            return x
        else:
            return self.sum(x)

    def sample(self, ctx: SamplingContext = None) -> th.Tensor:
        context_overhang = ctx.parent_indices.size(1) - self.sum.in_features
        assert context_overhang >= 0, f"context_overhang is negative! ({context_overhang})"
        if context_overhang:
            ctx.parent_indices = ctx.parent_indices[:, : -context_overhang]
        ctx = self.sum.sample(ctx=ctx)
        return super(GaussianMixture, self).sample(ctx=ctx)

    def moments(self):
        """Get the mean, variance and third central moment (unnormalized skew)"""
        if self._cached_moments is not None:
            return self._cached_moments
        dist_moments = [self.prod(self.pad_input(m), reduction=None) for m in self.base_leaf.moments()]
        weights = self.sum.weights
        if weights.dim() == 5:
            # Only in the Cspn case are the weights already log-normalized
            weights = weights.exp()
        else:
            weights = th.softmax(weights, dim=2)
        assert self.sum.weights.dim() == 5, "This isn't adopted to the 4-dimensional RatSpn weights yet"
        weights = weights.unsqueeze(2)
        # Weights is of shape [n, d, 1, ic, oc, r]
        # Create an extra dimension for the mean vector so all elements of the mean vector are multiplied by the same
        # weight for that feature and output channel.

        child_mean = dist_moments[0]
        # moments have shape [n, d, cardinality, ic, r]
        # Create an extra 'output channels' dimension, as the weights are separate for each output channel.
        child_mean.unsqueeze_(4)
        mean = child_mean * weights
        # mean has shape [n, d, cardinality, ic, oc, r]
        mean = mean.sum(dim=3)
        # mean has shape [n, d, cardinality, oc, r]
        moments = [mean]

        if len(dist_moments) >= 2:
            child_var = dist_moments[1]
            child_var.unsqueeze_(4)
            centered_mean = child_mean - mean.unsqueeze(4)
            var = child_var + centered_mean.pow(2)
            var = var * weights
            var = var.sum(dim=3)
            moments += [var]

            skew = 3 * centered_mean * child_var + centered_mean ** 3
            if len(dist_moments) >= 3:
                child_skew = dist_moments[2]
                child_skew.unsqueeze_(4)
                skew = skew + child_skew
            skew = skew * weights
            skew = skew.sum(dim=3)
            moments += [skew]

        self._cached_moments = moments
        return moments

    def _weighted_sum(self, x: th.Tensor):
        weights = self.sum.weights
        if weights.dim() == 5:
            # Only in the Cspn case are the weights already log-normalized
            weights = weights.exp()
        else:
            weights = F.softmax(weights, dim=2)

        # weights.unsqueeze(2) is of shape [n, d, 1, ic, oc, r]
        # The extra dimension is created so all elements of the gradient vectors are multiplied by the same
        # weight for that feature and output channel.
        return (x.unsqueeze(4) * weights.unsqueeze(2)).sum(dim=3)

    def gradient(self, x: th.Tensor, order: int):
        """Get the gradient up to the given order at the point x"""
        grads = self.base_leaf.gradient(x, order)
        grads = [self.prod(self.pad_input(g), reduction=None) for g in grads]
        grads = [self._weighted_sum(g) for g in grads]
        return grads

    def entropy_taylor_approx(self, components=3, reduction='mean'):
        mean, var, skew = self.moments()
        n, d, cardinality, oc, r = mean.shape
        # Gradients are all evaluated at the mean of the SPN
        mean_full_vec = mean.view(n, d * cardinality, oc, r)
        grads = self.gradient(mean_full_vec, order=components)
        log_p_mean = self(mean_full_vec, reduction=None)
        # clamp_at = -2
        # print(f"Percent of mean log probs under clamp threshold of {clamp_at}: "
              # f"{(log_p_mean < clamp_at).sum()/log_p_mean.numel():.5f}")
        # log_p_mean.clamp_(min=clamp_at)

        entropy = th.zeros(1).to(self._device)
        H_0 = th.zeros(1).to(self._device)
        H_2 = th.zeros(1).to(self._device)
        H_3 = th.zeros(1).to(self._device)
        if components >= 1:
            H_0 = - log_p_mean
            H_0 = H_0.sum(dim=2)
            if reduction == 'mean':
                H_0 = H_0.mean()
            entropy += H_0
            if components >= 2:
                grad, ggrad = grads[0:2]
                inv_mean_prob = (-log_p_mean).exp()
                inv_sq_mean_prob = (-2 * log_p_mean).exp()
                ggrad_log = -inv_sq_mean_prob * grad \
                            + inv_mean_prob * ggrad
                H_2 = - (ggrad_log * var) / 2
                H_2 = H_2.sum(dim=2)
                if reduction == 'mean':
                    H_2 = H_2.mean()
                entropy += H_2
                if components >= 3:
                    gggrad = grads[2]
                    inv_cub_mean_prob = (-3 * log_p_mean).exp()
                    gggrad_log: th.Tensor = 2 * inv_cub_mean_prob * grad \
                                               - 2 * inv_sq_mean_prob * ggrad \
                                               + inv_mean_prob * gggrad
                    H_3: th.Tensor = - (gggrad_log * skew) / 6
                    H_3 = H_3.sum(dim=2)
                    if reduction == 'mean':
                        H_3 = H_3.mean()
                    entropy += H_3

        return entropy, (H_0.detach_(), H_2.detach_(), H_3.detach_())

    def iterative_gmm_entropy_lb(self, reduction='mean'):
        """
            Calculate the entropy lower bound of the first-level mixtures.
            See "On Entropy Approximation for Gaussian Mixture Random Vectors" Huber et al. 2008, Theorem 2
        """
        log_gmm_weights: th.Tensor = self.sum.weights
        if log_gmm_weights.dim() == 4:
            # Only in the Cspn case are the weights already log-normalized
            log_gmm_weights = th.log_softmax(log_gmm_weights, dim=2)
        assert self.sum.weights.dim() == 5, "This isn't adopted to the 4-dimensional RatSpn weights yet"
        N, D, I, S, R = log_gmm_weights.shape
        # First sum layer after the leaves has weights of dim (N, D, I, S, R)
        # The entropy lower bound must be calculated for every sum node

        # bounded mean and variance
        # dist weights are of size (N, F, I, R)
        means, var = self.base_leaf.moments()
        _, F, _, _ = means.shape

        lb_ent_i = []
        for i in range(I):
            log_probs_i = []
            for j in range(I):
                summed_vars = var[:, :, [i], :] + var[:, :, [j], :]
                component_log_probs = -((means[:, :, [i], :] - means[:, :, [j], :]) ** 2) / (2 * summed_vars) - \
                                      0.5 * summed_vars.log() - math.log(math.sqrt(2 * math.pi))
                component_log_probs = self.prod(self.pad_input(component_log_probs))

                # Unsqueeze in output channel dimension so that the log_prob vector of each feature
                # is added to the weights of the S sum nodes of that feature.
                component_log_probs.unsqueeze_(dim=3)
                log_probs_i.append(log_gmm_weights[:, :, [j], :, :] + component_log_probs)
            log_probs_i = th.cat(log_probs_i, dim=2).logsumexp(dim=2, keepdim=True)
            lb_ent_i.append(log_gmm_weights[:, :, [i], :, :].exp() * log_probs_i)
        lb_ent = -th.cat(lb_ent_i, dim=2).sum(dim=2)

        if reduction == 'mean':
            lb_ent = lb_ent.mean()
        return lb_ent

    def gmm_entropy_lb(self, reduction='mean'):
        """
            Calculate the entropy lower bound of the first-level mixtures.
            See "On Entropy Approximation for Gaussian Mixture Random Vectors" Huber et al. 2008, Theorem 2
        """
        log_gmm_weights: th.Tensor = self.sum.weights
        if log_gmm_weights.dim() == 4:
            # Only in the Cspn case are the weights already log-normalized
            log_gmm_weights = th.log_softmax(log_gmm_weights, dim=2)
        assert self.sum.weights.dim() == 5, "This isn't adopted to the 4-dimensional RatSpn weights yet"
        N, D, I, S, R = log_gmm_weights.shape
        # First sum layer after the leaves has weights of dim (N, D, I, S, R)
        # The entropy lower bound must be calculated for every sum node

        # bounded mean and variance
        # dist weights are of size (N, F, I, R)
        means, var = self.base_leaf.moments()
        _, F, _, _ = means.shape

        repeated_means = means.repeat(1, 1, I, 1)
        var_outer_sum = var.unsqueeze(2) + var.unsqueeze(3)
        var_outer_sum = var_outer_sum.view(N, F, I**2, R)
        means_to_eval = means.repeat_interleave(I, dim=2)

        component_log_probs = -((means_to_eval - repeated_means) ** 2) / (2 * var_outer_sum) - \
                              0.5 * var_outer_sum.log() - math.log(math.sqrt(2 * math.pi))
        log_probs = self.prod(self.pad_input(component_log_probs))

        # Match sum weights to log probs
        w_j = log_gmm_weights.repeat(1, 1, I, 1, 1)
        # now [N, D, I^2, S, R]

        # Unsqueeze in output channel dimension so that the log_prob vector of each RV is added to the weights of
        # the S sum nodes of that RV.
        log_probs.unsqueeze_(dim=3)

        weighted_log_probs = w_j + log_probs
        lb_log_term = th.logsumexp(weighted_log_probs.view(N, D, I, I, S, R), dim=2)
        # lb_log_term is now [N, D, I, S, R], the same shape as log_gmm_weights

        gmm_ent_lb = -(log_gmm_weights.exp() * lb_log_term)
        gmm_ent_lb = gmm_ent_lb.sum(dim=2).sum(dim=1)
        # The entropies of all features can be summed up => [N, OC, R]
        if reduction == 'mean':
            gmm_ent_lb = gmm_ent_lb.mean()
        return gmm_ent_lb

    def __repr__(self):
        return f"GaussianMixture(in_features={self.in_features}, out_channels={self.out_channels}, dropout={self.dropout}, cardinality={self.cardinality}, out_shape={self.out_shape})"


def truncated_normal_(tensor, mean=0, std=0.1):
    """
    Truncated normal from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
