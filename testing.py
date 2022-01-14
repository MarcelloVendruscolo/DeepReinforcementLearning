import torch
import gym
from config import CartPole, Pong
from train import train
from evaluate import evaluate_policy
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # needed locally

ENV_CONFIGS = {
    'CartPole-v0': CartPole,
    'Pong-v0': Pong
}


def training(env):
    """
    function that is used to call the train() function inside train.py
    Parameter 'env' is the environment.
    """
    train(env)


def evaluating(env):
    """
    This function is written instead of the main() function inside evaluate.py
    path variable needs to be defined.
    """
    path = f'/Users/haradys/Documents/Data_Science_Master/RL/Project/models/{env}_best.pt'
    n_eval_episodes = 10
    render = True
    save_video = False
    env_name = env

    # Initialize environment and config
    env_config = ENV_CONFIGS[env]
    env = gym.make(env)

    if save_video:
        env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: True, force=True)

    # Load model from provided path.
    dqn = torch.load(path, map_location=torch.device('cpu'))
    dqn.eval()

    mean_return = evaluate_policy(dqn, env, env_config, env_name, n_eval_episodes, render=render and not save_video, verbose=True)
    print(f'The policy got a mean return of {mean_return} over {n_eval_episodes} episodes.')

    env.close()


if __name__ == "__main__":
    """
    In order to run this main, either 'Pong-v0' or 'CartPole-v0' needs to be uncommented,
    and either training or evaluating(with rendering) needs to be uncommented.
    """

    #env = 'CartPole-v0'
    env = 'Pong-v0'
    print('Start')
    training(env)
    print('End')
    #evaluating(env)