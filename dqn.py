"""
    In this file the DQN network with act method, Replay memory buffer and parameter optimization
    is implemented.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward, done)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.env_name = env_config['env_name']
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.epsilon_reduction_step = (self.eps_start - self.eps_end)/self.anneal_length
        self.n_actions = env_config["n_actions"]

        # if CartPole, then define layers as follows:
        if self.env_name == 'CartPole-v0':
            self.fc1 = nn.Linear(4, 256)
            self.fc2 = nn.Linear(256, self.n_actions)
        # else, pong
        else:
            self.obs_stack_size = env_config['obs_stack_size']

            self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=(4, 4), padding=(0, 0))
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
            self.fc1 = nn.Linear(in_features=3136, out_features=512)
            self.fc2 = nn.Linear(in_features=512, out_features=self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture.
           It is different for CartPole vs other atari games.
        """
        if self.env_name == 'CartPole-v0':
            x = self.relu(self.fc1(x))
            x = self.fc2(x)

        else:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)

        return x

    def reduce_epsilon(self):
        """
        Method that reduces self.eps_start linearly with self.epsilon_reduction
        length while it's greater that self.eps_end
        """
        if self.eps_start > self.eps_end:
            self.eps_start -= self.epsilon_reduction_step

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""

        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.

        def transform(ind):
            """Helper action mapping function for Pong action transformation from (0,1) to (2,3)"""
            return 2 if ind == 0 else 3

        n_observations = observation.size(0)
        predictions = self.forward(observation)  # predicted action values for each observation

        # first, deciding the range of actions for each game
        if self.env_name == 'CartPole-v0':
            low = 0
            high = self.n_actions
        else:
            # these actions are specific for the game Pong
            low = 2
            high = 4

        # first, when no exploration is needed when evaluating
        random_number = random.random()
        if exploit or random_number > self.eps_start:
            actions = torch.max(predictions, dim=1)[1]  # returns a tensor with indices of max values

            if self.env_name == 'Pong-v0':
                actions = torch.tensor(list(map(transform, actions))).to(device)

        # when exploration is needed
        else:
            actions = torch.randint(low=low, high=high, size=(n_observations, 1))  # return random actions for each obs
        return actions


def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.

    def transform(ind):
        """Helper action mapping function for Pong action transformation from (2,3) to (0,1)"""
        return 0 if ind == 2 else 1

    batch_size = dqn.batch_size
    if len(memory) < batch_size:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!

    sample = memory.sample(batch_size=batch_size)

    if dqn.env_name == 'CartPole-v0':
        observations = torch.row_stack(sample[0]).to(device)
        next_observations = torch.row_stack(sample[2]).to(device)

    else:
        observations = torch.cat(sample[0], dim=0).to(device)
        next_observations = torch.cat(sample[2], dim=0).to(device)

    actions = torch.tensor(sample[1], device=device)
    rewards = torch.tensor(sample[3], device=device)
    done = torch.tensor(sample[4], device=device)

    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.

    predictions = dqn.forward(observations).to(device)
    if dqn.env_name == 'Pong-v0':
        actions = torch.tensor(list(map(transform, actions))).to(device)

    q_values = torch.gather(predictions, dim=1, index=actions.unsqueeze(dim=1)).to(device)

    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!

    target_values = target_dqn.forward(next_observations).to(device)  # target q-values for next_states
    target_values = torch.max(target_values, dim=1)[0]  # choosing the max q-value for each next_state
    q_value_targets = rewards.unsqueeze(dim=1) + target_dqn.gamma*(torch.mul(done.unsqueeze(dim=1),
                                                                             target_values.unsqueeze(dim=1)))
    q_value_targets = q_value_targets.to(device)

    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets.squeeze())

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
