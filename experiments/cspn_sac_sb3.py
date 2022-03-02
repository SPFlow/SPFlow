import gym
import numpy as np
import sb3
from sb3 import CspnActor
import os

from cspn import CSPN, print_cspn_params

from stable_baselines3 import SAC

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cspn', action='store_true', help='Use a CSPN actor')
    parser.add_argument('--render_after_done', action='store_true', help='Don\' set this when running remotely')
    parser.add_argument('--timesteps', type=int, default=int(1e6), help='Total timesteps to train model.')
    parser.add_argument('--save_interval', type=int, default=int(1e4), help='Save model every save_interval timesteps.')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2', help='Gym environment to train on.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on. cpu or cuda.')
    parser.add_argument('--exp_name', type=str, default='test',
                        help='Experiment name. Will appear in name of saved model.')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the model to.')
    parser.add_argument('--model_path', type=str,
                        help='Path to the pretrained model.')
    parser.add_argument('--verbose', '-V', action='store_true', help='Output more debugging information when running.')
    # SAC arguments
    parser.add_argument('--ent_coef', type=float, default=0.1, help='Entropy temperature')
    args = parser.parse_args()

    # args.cspn = True
    # args.timesteps = 2000
    # args.save_interval = args.timesteps
    # args.verbose = True

    assert args.timesteps >= args.save_interval, "Total timesteps cannot be lower than save_interval!"
    assert args.timesteps % args.save_interval == 0, "save_interval must be a divisor of total timesteps."

    if args.save_dir:
        assert os.path.exists(args.save_dir), f"The save_dir doesn't exist! {args.save_dir}"
    if args.model_path:
        assert os.path.exists(args.model_path), f"The model_path doesn't exist! {args.model_path}"

    env = gym.make(args.env)
    if args.model_path:
        model = SAC.load(args.model_path)
        model_name = f"sac_loadedpretrained_{args.env}_{args.exp_name}"
    else:
        sac_kwargs = {
            'verbose': 2*args.verbose,
            'ent_coef': args.ent_coef,
            'learning_starts': 1000,
            'device': args.device,
        }
        if args.cspn:
            model = SAC("CspnPolicy", env, **sac_kwargs)
        else:
            model = SAC("MlpPolicy", env, **sac_kwargs)
        model_name = f"sac_{'cspn' if args.cspn else 'mlp'}_{args.env}_{args.exp_name}"

    if isinstance(model.actor, CspnActor):
        print_cspn_params(model.actor.cspn)
    else:
        print(f"Actor MLP has {sum(p.numel() for p in model.actor.parameters() if p.requires_grad)} parameters.")
    num_epochs = int(args.timesteps // args.save_interval)
    for i in range(num_epochs):
        model.learn(total_timesteps=args.save_interval, log_interval=1 if args.verbose else 4)
        model.save(os.path.join(args.save_dir, f"{model_name}_{(i+1)*args.save_interval}steps"))

    if args.render_after_done:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()
