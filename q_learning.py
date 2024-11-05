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

    def __init__(self, state: GamblerState, action: int, reward: float, next_state: GamblerState):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    def __repr__(self) -> str:
        return f"({self.state}, {self.action}, {self.reward}, {self.next_state})"

class TransitionBuffer:
    """A buffer that stores previous transitions and supports sampling for training."""
    transitions: list[Transition]
    index: int
    rng: Generator

    def __init__(self, rng: Generator):
        self.transitions = []
        self.index = 0
        self.rng = rng

    def insert(self, transitions: list[Transition]):
        """Adds transitions to the buffer."""
        for transition in transitions:
            if len(self.transitions) < BUFFER_SIZE:
                self.transitions.append(transition)
            else:
                self.transitions[self.index] = transition
                self.index = (self.index + 1) % len(self.transitions)

    def sample(self, count: int) -> list[Transition]:
        """Returns random transitions from the buffer."""
        assert 1 <= count and count <= len(self.transitions)
        indices = self.rng.choice(len(self.transitions), size=count, replace=False)
        sample = []
        for index in indices:
            sample.append(self.transitions[index])
        return sample

    def __len__(self) -> int:
        """Returns the number of transitions in the buffer."""
        return len(self.transitions)

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

    while not state.done:
        action = None
        if rng.random() < explore_factor:
            # Take random 'explore' action
            actions = np.nonzero(state.get_action_mask().numpy())[0] + 1
            action = rng.choice(actions)
        else:
            # Take 'exploit' action
            with torch.no_grad():
                values = model.forward(state.get_observation())
                values[~state.get_action_mask()] = -np.inf
                action = int(torch.argmax(values).item() + 1)

        reward, next_state = env.step(state, action)
        transitions.append(Transition(state, action, reward, next_state))
        state = next_state

    return transitions

def train(env: GamblerGame, seed: int):
    """Trains a Deep Q-Learning agent on the gambler Markov decision process."""
    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(rng.integers(SEED_RANGE[0], SEED_RANGE[1]))

    # Actions are sampled from the policy network and value targets are computed
    # using the target network. Keeping the target network fixed for several
    # training iterations at a time stabilizes the training.
    policy_network = ValueNetwork()
    target_network = ValueNetwork()
    target_network.load_state_dict(policy_network.state_dict())

    # Initialize training
    transitions = TransitionBuffer(rng)
    explore_factor = INITIAL_EXPLORE