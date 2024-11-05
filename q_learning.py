import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import GamblerGame, GamblerState

# Training parameters
STATE_SIZE: int = 21
ACTION_SIZE: int = STATE_SIZE - 2
HIDDEN_SIZE: int = 64
INITIAL_EXPLORE: float = 1
DISCOUNT_RATE: float = 0.97
LEARNING_RATE: float = 0.01

class ValueNetwork(nn.Module):
    """
    A Q-value neural network that predicts the discounted value for each
    action at a state.
    """

def train(env: GamblerGame, seed: int):
    """Trains a Deep Q-Learning agent on the gambler Markov decision process."""
    print(env)
    print(seed)