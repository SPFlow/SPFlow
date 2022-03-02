import gym
import numpy as np
import sb3
import os

from stable_baselines3 import SAC

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cspn', action='store_true', help='Use a CSPN actor')
    parser.add_argument('--render_after_done', action='store_true', help='Don\' set this when running remotely')
    parser.add_argument('--timesteps', type=int, default=1e6, help='Total timesteps to train model.')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2', help='Gym environment to train on.')
    parser.add_argument('--exp_name', type=str, default='test',
                        help='Experiment name. Will appear in name of saved model.')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the model to.')
    parser.add_argument('--model_path', type=str,
                        help='Path to the pretrained model.')
    args = parser.parse_args()

    if args.save_dir:
        assert os.path.exists(args.save_dir), f"The save_dir doesn't exist! {args.save_dir}"
    if args.model_path:
        assert os.path.exists(args.model_path), f"The model_path doesn't exist! {args.model_path}"

    env = gym.make(args.env)
    if args.model_path:
        model = SAC.load(args.model_path)
        model_name = f"sac_loadedpretrained_{args.env}_{args.exp_name}"
    else:
        if args.cspn:
            model = SAC("CspnPolicy", env, verbose=1)
        else:
            model = SAC("MlpPolicy", env, verbose=1)
        model_name = f"sac_{'cspn' if args.cspn else 'mlp'}_{args.env}_{args.exp_name}"
    model.learn(total_timesteps=args.timesteps, log_interval=4)
    model.save(os.path.join(args.save_dir, model_name))

    if args.render_after_done:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()
