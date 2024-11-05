import numpy as np
import torch.nn as nn
from typing import Optional

from environment import GamblerGame, GamblerState

class Evaluation:
    """
    Evaluates a policy in the gambler Markov decision process by running random
    episodes and computing the mean reward.
    """
    env: GamblerGame
    episodes: int

    def __init__(self, env: GamblerGame, episodes: int):
        self.env = env
        self.episodes = episodes

    def evaluate_q_table(self, q_table: np.ndarray, episodes: Optional[int] = None) -> float:
        """Evaluates a Q-table policy."""
        return 0

    def evaluate_q_network(self, model: nn.Module, episodes: Optional[int] = None) -> float:
        """Evaluates a Q-value neural network policy."""
        return 0

    def evaluate_optimal(self, episodes: Optional[int] = None) -> float:
        """Evaluates the optimal policy."""
        return 0

    def _get_q_table_action(self, q_table: np.ndarray, state: GamblerState) -> int:
        """Queries an explicit Q-table for an action at a state."""
        values = q_table[state.get_index()].copy()
        values[~state.get_action_mask().numpy()] = -np.inf
        return int(np.argmax(values))

    def _get_q_table(self, model: nn.Module) -> np.ndarray:
        """
        Queries the model at each state to calculate the Q-table. For environments with large
        state spaces or continuous states, storing the Q-table explicitly is intractable.
        The gambler game has a small state space so we can represent the Q-table as a
        2-dimensional array. Note that we only use the Q-table for efficient model evaluation.
        """
        q_table = np.zeros((self.env.get_state_size(), self.env.get_action_size()))
        for state_index in range(1, self.env.get_state_size() - 1):
            state = self.env.create_state(state_index)
            print(state)

        return np.array([])

    def _get_optimal_action(self, state: GamblerState) -> int:
        """Computes the optimal action at a state based on the win probability."""
        if self.env.win_prob >= 0.5:
            return 0
        return min(state.wealth, self.env.target_wealth - state.wealth) - 1