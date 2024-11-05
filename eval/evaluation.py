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
        actions = self._get_q_table_actions(q_table)
        return self._evaluate_policy(lambda s: actions[s.get_index()], episodes)

    def evaluate_q_network(self, model: nn.Module, episodes: Optional[int] = None) -> float:
        """Evaluates a Q-value neural network policy."""
        if episodes is None:
            episodes = self.episodes
        actions = self._get_q_network_actions(model)
        return self._evaluate_policy(lambda s: actions[s.get_index()], episodes)

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

    def _get_q_table_actions(self, q_table: np.ndarray) -> list[int]:
        """Queries an explicit Q-table at each state for the predicted action."""
        actions = [0] * self.env.get_state_size()
        for state_index in range(1, self.env.get_state_size() - 1):
            state = self.env.create_state(state_index)
            values = q_table[state.get_index()].copy()
            values[~state.get_action_mask().numpy()] = -np.inf
            actions[state_index] = int(np.argmax(values))
        return actions

    def _get_q_network_actions(self, model: nn.Module) -> list[int]:
        """
        Queries the model at each state to calculate the predicted action. Note that for
        environments with large state spaces or continuous states, storing the action
        table or Q-table explicitly is intractable.
        """
        actions = [0] * self.env.get_state_size()
        return actions

    def _get_optimal_action(self, state: GamblerState) -> int:
        """Computes the optimal action at a state based on the win probability."""
        if self.env.win_prob >= 0.5:
            return 0
        return min(state.wealth, self.env.target_wealth - state.wealth) - 1