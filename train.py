"""
    In this file we define the training of the agent, and a function for plotting the
    mean rewards per episode.
"""

import argparse
import copy

import gym
import torch
import matplotlib.pyplot as plt

from config import CartPole, Pong
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': CartPole,
    'Pong-v0': Pong
}


def plotting(env_name, n_episodes, mean_return, show=False):
    """Creating a plot of the mean return for each evaluation step
        env_name: name of environment
        n_episodes: number of episodes
        mean_return: list of the mean returns"""

    fig, ax = plt.subplots(figsize=(8, 6))
    name = f'{env_name}'
    ax.set(xlabel='Episode', ylabel='Mean return', title=f'Mean return of evaluation for {name}')
    ax.set_xlim(left=1, right=n_episodes)
    x, y = zip(*mean_return)
    ax.scatter(x, y, color='blue')
    path = '/content/drive/MyDrive/Colab_Notebooks/Project/plots'
    plt.savefig(f'{path}/{name}.png')
    if show:
        plt.show()

# Ex.
# env: 'CartPole-v0'
# env: 'Pong-v0'


def train(env):
    # Initialize environment and config.
    env_config = ENV_CONFIGS[env]
    env_name = env_config['env_name']
    env = gym.make(env)

    if env_name != 'CartPole-v0':
        env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    target_dqn = copy.deepcopy(dqn).to(device)  # copy the dqn parameters using deepcopy

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    n_episodes = env_config['n_episodes']
    steps = 0  # number of steps taken during the entire training
    evaluate_freq = 25  # How often to run evaluation
    evaluation_episodes = 4  # Number of evaluation episodes
    mean_return_train = []  # list that will contain mean return for each evaluation phase
    k = 4  # take a new action in every kth frame instead of every frame

    for episode in range(n_episodes):
        done = False

        obs = preprocess(env.reset(), env=env_name).unsqueeze(0)  # size = (1,84,84) for Pong

        # creating a stack of 'obs_stack_size' observations from obs
        if env_name != 'CartPole-v0':
            obs_stack = torch.cat(dqn.obs_stack_size * [obs]).to(device)  # size = (4,84,84)

        while not done:
            dqn.reduce_epsilon()  # method that reduces epsilon automatically if needed

            # first if the env is CartPole
            if env_name == 'CartPole-v0':
                # TODO: Get action from DQN.
                action = dqn.act(obs).item()

                # Act in the true environment.
                next_obs, reward, done, info = env.step(action)

                # Preprocess incoming observation.
                next_obs = preprocess(next_obs, env=env_name).unsqueeze(0)

                # TODO: Add the transition to the replay memory. Remember to convert
                #       everything to PyTorch tensors!
                # why 1-done is because non-terminal states has done=0, but we want to compute q-value for those states
                memory.push(obs, action, next_obs, reward, 1-int(done))

                if not done:
                    obs = next_obs

            # pong
            else:

                if steps % k == 0:
                    action = dqn.act(obs_stack.unsqueeze(0)).item()
                next_obs, reward, done, info = env.step(action)
                next_obs_stack = preprocess(next_obs, env=env_name).unsqueeze(0)
                next_obs_stack = torch.cat((obs_stack[1:, :, ...], next_obs_stack), dim=0).to(device)
                memory.push(obs_stack.unsqueeze(0), action, next_obs_stack.unsqueeze(0), reward, 1-int(done))

                if not done:
                    obs_stack = next_obs_stack

            steps += 1  # increment steps

            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            train_frequency = env_config['train_frequency']
            if steps % train_frequency == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            target_update_frequency = env_config["target_update_frequency"]
            if steps % target_update_frequency == 0:
                target_dqn = copy.deepcopy(dqn)

        # Evaluate the current agent.
        if episode % evaluate_freq == 0:
            mean_return = evaluate_policy(dqn=dqn, env=env, env_config=env_config, env_name=env_name,
                                          n_episodes=evaluation_episodes, render=False)
            mean_return_train.append((episode, mean_return))
            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                path = '/content/drive/MyDrive/Colab_Notebooks/Project/models'
                torch.save(dqn, f'{path}/{env_name}_best_train2.pt')

                # each time a best model is found, a plot is made.
                plotting(env_name=env_name, n_episodes=env_config['n_episodes'], mean_return=mean_return_train,
                         show=False)

    # plotting the whole training
    print(f'Best performance in final step! {best_mean_return}')
    plotting(env_name=env_name, n_episodes=env_config['n_episodes'], mean_return=mean_return_train, show=True)

    # Close environment after training is completed.
    env.close()
