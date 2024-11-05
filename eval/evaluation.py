import numpy as np
from numpy.random import Generator
import torch.nn as nn
from typing import Callable, Optional

from environment import GamblerGame, GamblerState

class Evaluation:
    """
    Evaluates a policy in the gambler Markov decision process by running random
    episodes and computing the mean reward.
    """
    env: GamblerGame
    episodes: int
    rng: Generator

    def __init__(self, env: GamblerGame, episodes: int, seed: int):
        self.env = env
        self.episodes = episodes
        self.rng = np.random.default_rng(seed)

    def evaluate_q_table(self, q_table: np.ndarray, episodes: Optional[int] = None) -> float:
        """Evaluates a Q-table policy."""
        if episodes is None:
            episodes = self.episodes
        return self._evaluate_policy(
            lambda state: self._get_q_table_action(q_table, state),
            episodes,
        )

    def evaluate_q_network(self, model: nn.Module, episodes: Optional[int] = None) -> float:
        """Evaluates a Q-value neural network policy."""
        q_table = self._get_q_table(model)
        return self.evaluate_q_table(q_table, episodes)

    def evaluate_optimal(self, episodes: Optional[int] = None) -> float:
        """Evaluates the optimal policy."""
        if episodes is None:
            episodes = self.episodes
        return self._evaluate_policy(self._get_optimal_action, episodes)

    def _evaluate_policy(self, policy_fn: Callable[[GamblerState], int], episodes: int) -> float:
        """Evaluates a policy function that takes a state and returns an action."""
        total_reward = 0
        for e in range(episodes):
            state = self.env.reset()
            while not state.done:
                action = policy_fn(state)
                reward, state = self.env.step(state, action)
                total_reward += reward
        return total_reward / episodes

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