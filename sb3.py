import stable_baselines3 as sb3
import gym
from stable_baselines3.sac.policies import SACPolicy
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
import numpy as np

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule

from cspn import CspnConfig, CSPN
from distributions import RatNormal


class CspnActor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            features_extractor: nn.Module,
            features_dim: int,
            R: int,
            D: int,
            I: int,
            S: int,
            dropout: float,
            sum_layers: int,
            dist_layers: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            **kwargs
    ):
        super(CspnActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)

        config = CspnConfig()
        config.F_cond = (features_dim,)
        config.C = 1
        config.nr_feat_layers = 0
        config.fc_sum_param_layers = sum_layers
        config.fc_dist_param_layers = dist_layers
        config.F = action_dim
        config.R = R
        config.D = D if D is not None else int(np.log2(action_dim))
        config.I = I
        config.S = S
        config.dropout = dropout
        config.leaf_base_class = RatNormal
        # TODO are actions tanh_bounded somewhere else? What about the correction term?
        config.leaf_base_kwargs = {'tanh_bounds': (-1.0, 1.0)}
        # config.leaf_base_kwargs = {'min_mean': 0.0, 'max_mean': 1.0}
        if False:
            config.leaf_base_kwargs['min_sigma'] = 0.1
            config.leaf_base_kwargs['max_sigma'] = 1.0
        self.config = config
        self.cspn = CSPN(config)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                log_std_init=self.log_std_init,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the action is squashed
        features = self.extract_features(obs)
        return self.cspn.sample(condition=features, is_mpe=deterministic)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # return action and entropy
        features = self.extract_features(obs)
        action = self.cspn.sample(condition=features, is_mpe=False).squeeze(0)
        entropy = self.cspn.vi_entropy_approx(sample_size=5, condition=None)
        return action, entropy

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


class CspnPolicy(SACPolicy):
    """
    Policy class (with a CSPN actor and an MLP critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param critic_arch: The specification of the value networks.
    :param activation_fn: Activation function
    :param log_std_init: Initial value for the log standard deviation
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            log_std_init: float = -3,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            **kwargs
    ):
        super(SACPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [256, 256]

        if isinstance(net_arch, dict) and 'pi' in net_arch.keys():
            warnings.warn("CspnPolicy ignores pi net_arch settings, as the Cspn needs different configuration.")
        _, critic_arch = get_actor_critic_arch(net_arch)

        self.critic_arch = critic_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.actor_kwargs.update(
            {
                'R': 3,
                'D': None,  # Set to None to make D == log2(action_dim)
                'I': 5,
                'S': 5,
                'dropout': 0.0,
                'sum_layers': 2,
                'dist_layers': 2,
            }
        )

        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CspnActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CspnActor(**actor_kwargs).to(self.device)


register_policy("CspnPolicy", CspnPolicy)
