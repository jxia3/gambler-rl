import numpy as np
from numpy.random import Generator
from typing import Optional

from environment import GamblerGame, GamblerState

# The inclusive integer range for sampling PyTorch seeds
SEED_RANGE: tuple[int, int] = (0, 1_000_000_000)

# Training parameters
STATE_SIZE: int = 11
ACTION_SIZE: int = 2
DISCOUNT_RATE: float = 0.98
LEARNING_RATE: float = 0.001

INITIAL_EXPLORE: float = 1
EXPLORE_DECAY: float = 0.9994
MIN_EXPLORE: float = 0.05

EPISODES: int = 1000
LOG_INTERVAL: int = 100

class Transition:
    state: int
    action: int
    reward: float
    next_state: Optional[int]

    def __init__(self, state: int, action: int, reward: float, next_state: Optional[int]):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

def run_rollout(
    env: GamblerGame,
    q_table: np.ndarray,
    explore_factor: float,
    rng: Generator,
) -> list[Transition]:
    transitions = []
    state = env.reset()

    while not state.done:
        print("at:", state)
        action = None
        if rng.random() < explore_factor:
            actions = np.nonzero(state.get_action_mask().numpy())[0]
            action = rng.choice(actions)
            print("explore action", actions, action)
        else:
            state_index = np.argmax(state.get_observation().numpy())
            print("exploit action", state_index, q_table[state_index])
        break

    return transitions

def train(env: GamblerGame, seed: int):
    """Trains a tabular Q-Learning agent on the gambler Markov decision process."""
    rng = np.random.default_rng(seed=seed)
    q_table = np.zeros((STATE_SIZE, ACTION_SIZE))
    explore_factor = INITIAL_EXPLORE

    for episode in range(1, EPISODES + 1):
        trajectory = run_rollout(env, q_table, explore_factor, rng)