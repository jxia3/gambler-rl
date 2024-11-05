import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import GamblerGame, GamblerState

# Training parameters
STATE_SIZE: int = 21
ACTION_SIZE: int = STATE_SIZE - 2
HIDDEN_SIZE: int = 64
DISCOUNT_RATE: float = 0.97
LEARNING_RATE: float = 0.01

INITIAL_EXPLORE: float = 1
EXPLORE_DECAY: float = 0.999
MIN_EXPLORE: float = 0.1
BUFFER_SIZE: int = 10000
BATCH_SIZE: int = 256

class ValueNetwork(nn.Module):
    """
    A Q-value neural network with 1 hidden layer that predicts the discounted
    value for each action at a state.
    """

    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.hidden = nn.Linear(STATE_SIZE, HIDDEN_SIZE)
        self.output = nn.Linear(HIDDEN_SIZE, ACTION_SIZE)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.hidden(state)
        out = F.relu(out)
        out = self.output(state)
        return out

def train(env: GamblerGame, seed: int):
    """Trains a Deep Q-Learning agent on the gambler Markov decision process."""
    print(env)
    print(seed)

    model = ValueNetwork()
    print(model)