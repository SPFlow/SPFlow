import gym
import numpy as np
import sb3
from sb3 import CspnActor, CspnSAC
import os
import platform

import torch.nn as nn

from cspn import CSPN, print_cspn_params

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cspn', action='store_true', help='Use a CSPN actor')
    parser.add_argument('--render_after_done', action='store_true', help='Don\' set this when running remotely')
    parser.add_argument('--timesteps', type=int, default=int(1e6), help='Total timesteps to train model.')
    parser.add_argument('--save_interval', type=int, help='Save model every save_interval timesteps.')
    parser.add_argument('--log_interval', type=int, default=4, help='Log interval')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2', help='Gym environment to train on.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on. cpu or cuda.')
    parser.add_argument('--exp_name', type=str, default='test',
                        help='Experiment name. Will appear in name of saved model.')
    parser.add_argument('--save_dir', type=str, default='../../cspn_rl_experiments',
                        help='Directory to save the model to.')
    parser.add_argument('--tensorboard_dir', '-tb', type=str, default='../../cspn_rl_experiments/tb',
                        help='Directory to save the model to.')
    parser.add_argument('--model_path', type=str,
                        help='Path to the pretrained model.')
    parser.add_argument('--verbose', '-V', action='store_true', help='Output more debugging information when running.')
    # SAC arguments
    parser.add_argument('--ent_coef', type=float, default=0.1, help='Entropy temperature')
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--learning_starts', type=int, default=1000,
                        help='Nr. of steps to act randomly in the beginning.')
    # CSPN arguments
    parser.add_argument('--repetitions', '-R', type=int, default=5, help='Number of parallel CSPNs to learn at once. ')
    parser.add_argument('--cspn_depth', '-D', type=int,
                        help='Depth of the CSPN. If not provided, maximum will be used (ceil of log2(inputs)).')
    parser.add_argument('--num_dist', '-I', type=int, default=5, help='Number of Gauss dists per pixel.')
    parser.add_argument('--num_sums', '-S', type=int, default=5, help='Number of sums per RV in each sum layer.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout to apply')
    parser.add_argument('--no_relu', action='store_true',
                        help='Don\'t use inner ReLU activations in the layers providing '
                             'the CSPN parameters from the conditional.')
    parser.add_argument('--feat_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN feature layers.')
    parser.add_argument('--sum_param_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN sum param layers.')
    parser.add_argument('--dist_param_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN dist param layers.')
    args = parser.parse_args()

    if not args.save_interval:
        args.save_interval = args.timesteps

    if args.timesteps == 0:
        learn = False
    else:
        learn = True
        assert args.timesteps >= args.save_interval, "Total timesteps cannot be lower than save_interval!"
        assert args.timesteps % args.save_interval == 0, "save_interval must be a divisor of total timesteps."

    if args.save_dir:
        assert os.path.exists(args.save_dir), f"The save_dir doesn't exist! {args.save_dir}"
    if args.model_path:
        assert os.path.exists(args.model_path), f"The model_path doesn't exist! {args.model_path}"
    if args.tensorboard_dir:
        assert os.path.exists(args.tensorboard_dir), f"The tensorboard_dir doesn't exist! {args.tensorboard_dir}"

    env_name = 'HalfCheetah-v2'
    results_dir = f"{platform.node()}_SAC_{args.exp_name}_{env_name}"
    results_path = os.path.join(args.save_dir, results_dir)
    for d in [results_path]:
        if not os.path.exists(d):
            os.makedirs(d)

    env = make_vec_env(
        env_id=env_name,
        n_envs=1,
        monitor_dir=results_path,
        # monitor_dir=os.path.join(results_path, f"log_{args.exp_name}.txt"),
        # vec_env_cls=SubprocVecEnv,
        # vec_env_kwargs={'start_method': 'fork'},
    )

    if args.model_path:
        model = SAC.load(args.model_path, env)
        model_name = f"sac_loadedpretrained_{args.env}_{args.exp_name}"
    else:
        sac_kwargs = {
            'verbose': 2*args.verbose,
            'ent_coef': args.ent_coef,
            'learning_starts': args.learning_starts,
            'device': args.device,
            'tensorboard_log': args.tensorboard_dir,
            'learning_rate': args.learning_rate,
        }
        if args.cspn:
            cspn_args = {
                'R': args.repetitions,
                'D': args.cspn_depth,
                'I': args.num_dist,
                'S': args.num_sums,
                'dropout': args.dropout,
                'feat_layers': args.feat_layers,
                'sum_param_layers': args.sum_param_layers,
                'dist_param_layers': args.dist_param_layers,
                'cond_layers_inner_act': nn.Identity if args.no_relu else nn.ReLU,
            }
            sac_kwargs['policy_kwargs'] = {'cspn_args': cspn_args}
            model = CspnSAC("CspnPolicy", env, **sac_kwargs)
        else:
            model = SAC("MlpPolicy", env, **sac_kwargs)
        model_name = f"sac_{'cspn' if args.cspn else 'mlp'}_{args.env}_{args.exp_name}"

    print(model.actor)
    print(model.critic)
    if isinstance(model.actor, CspnActor):
        print_cspn_params(model.actor.cspn)
    else:
        print(f"Actor MLP has {sum(p.numel() for p in model.actor.parameters() if p.requires_grad)} parameters.")
    if learn:
        num_epochs = int(args.timesteps // args.save_interval)
        for i in range(num_epochs):
            model.learn(
                total_timesteps=args.save_interval,
                log_interval=args.log_interval,
                reset_num_timesteps=False,
                tb_log_name=results_dir,
            )
            model.save(os.path.join(results_path, f"{model_name}_{(i+1)*args.save_interval}steps"))

    if args.render_after_done:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()
