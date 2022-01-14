"""
 This file contains one function that takes an environment and state as input, process the state
 and returns a tensor as output.
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v0']:
        return torch.tensor(obs, device=device).float()
    else:
        obs = obs/255.0  # rescaling observations from 0-255 to 0-1
        return torch.tensor(obs, device=device).float()
