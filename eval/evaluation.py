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

    def _get_optimal_action(self, state: GamblerState) -> int:
        """Computes the optimal action at a gambler game state."""
        if self.env.win_prob >= 0.5:
            return 0
        return min(state.wealth, self.env.target_wealth - state.wealth) - 1