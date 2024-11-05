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

    def __repr__(self) -> str:
        return f"({self.state}, {self.action}, {self.reward}, {self.next_state})"

def run_rollout(
    env: GamblerGame,
    q_table: np.ndarray,
    explore_factor: float,
    rng: Generator,
) -> list[Transition]:
    transitions = []
    state = env.reset()
    observation = state.get_observation().numpy()

    while not state.done:
        state_index = int(np.argmax(observation))
        action_mask = state.get_action_mask()
        action = None
        if rng.random() < explore_factor:
            actions = np.nonzero(action_mask)[0]
            action = rng.choice(actions)
        else:
            values = q_table[state_index].copy()
            values[~action_mask] = -np.inf
            action = int(np.argmax(values))

        reward, next_state = env.step(state, action)
        next_observation = next_state.get_observation().numpy()
        next_index = None
        if not next_state.done:
            next_index = int(np.argmax(next_observation))

        transitions.append(Transition(state_index, action, reward, next_index))
        state = next_state
        observation = next_observation

    return transitions

def train(env: GamblerGame, seed: int):
    """Trains a tabular Q-Learning agent on the gambler Markov decision process."""
    rng = np.random.default_rng(seed=seed)
    q_table = np.zeros((STATE_SIZE, ACTION_SIZE), dtype=np.float32)
    explore_factor = INITIAL_EXPLORE

    for episode in range(1, EPISODES + 1):
        trajectory = run_rollout(env, q_table, explore_factor, rng)
        print(trajectory)