import gym
import numpy as np
import sb3
from sb3 import CspnActor
import os

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
    parser.add_argument('--model_path', type=str,
                        help='Path to the pretrained model.')
    parser.add_argument('--verbose', '-V', action='store_true', help='Output more debugging information when running.')
    # SAC arguments
    parser.add_argument('--ent_coef', type=float, default=0.1, help='Entropy temperature')
    # CSPN arguments
    parser.add_argument('--repetitions', '-R', type=int, default=5, help='Number of parallel CSPNs to learn at once. ')
    parser.add_argument('--cspn_depth', '-D', type=int,
                        help='Depth of the CSPN. If not provided, maximum will be used (ceil of log2(inputs)).')
    parser.add_argument('--num_dist', '-I', type=int, default=5, help='Number of Gauss dists per pixel.')
    parser.add_argument('--num_sums', '-S', type=int, default=5, help='Number of sums per RV in each sum layer.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout to apply')
    parser.add_argument('--feat_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN feature layers.')
    parser.add_argument('--sum_param_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN sum param layers.')
    parser.add_argument('--dist_param_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN dist param layers.')
    parser.add_argument('--plot_vi_log', action='store_true',
                        help='Collect information from variational inference entropy '
                             'approx. and plot it at the end of the epoch.')
    args = parser.parse_args()

    # args.cspn = True
    # args.timesteps = 2000
    # args.save_interval = args.timesteps
    # args.verbose = True

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

    args.save_dir = os.path.join(args.save_dir, f"results_{args.exp_name}")
    for d in [args.save_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    env = make_vec_env(
        env_id='HalfCheetah-v2',
        n_envs=1,
        monitor_dir=args.save_dir,
        # monitor_dir=os.path.join(args.save_dir, f"log_{args.exp_name}.txt"),
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
            'learning_starts': 1000,
            'device': args.device,
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
                'log_vi_ent_approx': args.plot_vi_log,
            }
            sac_kwargs['policy_kwargs'] = {'cspn_args': cspn_args}
            model = SAC("CspnPolicy", env, **sac_kwargs)
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
            model.learn(total_timesteps=args.save_interval, log_interval=args.log_interval)
            model.save(os.path.join(args.save_dir, f"{model_name}_{(i+1)*args.save_interval}steps"))

    if args.render_after_done:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()
