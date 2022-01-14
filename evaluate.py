"""
    In this file we define the agent evaluation.
"""

import argparse

import gym
import torch
from time import sleep
import config
from utils import preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0'])
parser.add_argument('--path', type=str, help='Path to stored DQN model.')
parser.add_argument('--n_eval_episodes', type=int, default=1, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--render', dest='render', action='store_true', help='Render the environment.')
parser.add_argument('--save_video', dest='save_video', action='store_true', help='Save the episodes as video.')
parser.set_defaults(render=False)
parser.set_defaults(save_video=False)


# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole,
    'Pong-v0': config.Pong
}


def evaluate_policy(dqn, env, env_config, env_name, n_episodes, render=False, verbose=False):
    """Runs {n_episodes} episodes to evaluate current policy."""

    total_return = 0
    if env_name == 'Pong-v0':
        env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True,
                                              frame_skip=1, noop_max=30)

    for i in range(n_episodes):
        obs = preprocess(env.reset(), env=env_name).unsqueeze(0)

        if env_name != 'CartPole-v0':
            obs_stack = torch.cat(dqn.obs_stack_size * [obs]).to(device)

        done = False
        episode_return = 0

        while not done:
            if render:
                sleep(0.05)
                env.render()

            if env_name == 'CartPole-v0':
                action = dqn.act(obs, exploit=True).item()
                obs, reward, done, info = env.step(action)
                obs = preprocess(obs, env=env_name).unsqueeze(0)

            else:
                action = dqn.act(obs_stack.unsqueeze(0), exploit=True).item()
                n_obs, reward, done, info = env.step(action)
                n_obs = preprocess(n_obs, env=env_name).unsqueeze(0)
                obs_stack = torch.cat((obs_stack[1:, :, ...], n_obs), dim=0)

            episode_return += reward

        total_return += episode_return

        if verbose:
            print(f'Finished episode {i+1} with a total return of {episode_return}')

    return total_return / n_episodes


# This main() function is unused since we have added a new file testing.py
if __name__ == '__main__':
    args = parser.parse_args()
    env = 'CartPole-v0'
    path = '/models/CartPole-v0_best.pt'
    n_eval_episodes = 50
    render = True
    save_video = True

    # Initialize environment and config
    env_config = ENV_CONFIGS[env]
    env = gym.make(env)

    if save_video:
        env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: True, force=True)

    # Load model from provided path.
    dqn = torch.load(args.path, map_location=torch.device('cpu'))
    dqn.eval()

    mean_return = evaluate_policy(dqn, env, env_config, args, args.n_eval_episodes,
                                  render=args.render and not args.save_video, verbose=True)
    print(f'The policy got a mean return of {mean_return} over {args.n_eval_episodes} episodes.')

    env.close()