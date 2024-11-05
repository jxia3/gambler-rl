import numpy as np
from numpy.random import Generator
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import GamblerGame, GamblerState

# The inclusive integer range for sampling PyTorch seeds
SEED_RANGE: tuple[int, int] = (0, 1_000_000_000)

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
        out = self.output(out)
        return out

class Transition:
    """
    A tuple (s_t, a_t, r_t, s_{t + 1}) tracking a state transition in the Markov
    decision process. Transitions are stored in a buffer used for training.
    """
    state: GamblerState
    action: int
    reward: float
    next_state: GamblerState

def run_rollout(
    env: GamblerGame,
    model: ValueNetwork,
    explore_factor: float,
    rng: Generator,
) -> list[Transition]:
    """
    Simulates a trajectory in the Markov decision process with the Q-value network.
    Random 'explore' actions are chosen with probability explore_factor and otherwise
    'exploit' actions with the maximum predicted value are chosen.
    """
    transitions = []
    state = env.reset()

    return transitions

def train(env: GamblerGame, seed: int):
    """Trains a Deep Q-Learning agent on the gambler Markov decision process."""
    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(rng.integers(SEED_RANGE[0], SEED_RANGE[1]))
    model = ValueNetwork()

    print(model)
    state = env.reset()
    print(state)
    print(state.get_observation())
    print(model.forward(state.get_observation()))