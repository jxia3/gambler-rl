import numpy as np
from numpy.random import Generator
import torch
from typing import Optional

SEED_RANGE = (0, 1_000_000_000)

class GamblerState:
    """
    A state in the gambler Markov decision process that contains the current
    wealth and the random generator for the episode.
    """
    target_wealth: int
    wealth: int
    done: bool
    rng: Generator

    def __init__(self, target_wealth: int, wealth: int, done: bool, rng: Generator):
        self.target_wealth = target_wealth
        self.wealth = wealth
        self.done = done
        self.rng = rng

    def get_observation(self) -> torch.Tensor:
        """Encodes the player's current wealth as a one-hot vector."""
        observation = torch.zeros(self.target_wealth + 1, dtype=torch.int32)
        observation[self.wealth] = 1
        return observation

    def get_action_mask(self) -> torch.Tensor:
        """Encodes the legal actions given the player's current wealth."""
        action_mask = torch.zeros(self.target_wealth - 1, dtype=torch.int32)
        action_mask[:self.wealth] = 1
        return action_mask

    def __repr__(self) -> str:
        """Formats the state as a string."""
        return f"GamblerState(target_wealth={self.target_wealth}, wealth={self.wealth})"

class GamblerGame:
    """
    An implementation of the gambler betting game as a Markov decision process.
    The player starts with a random initial wealth and bets an integer amount at
    each time step to reach TARGET_WEALTH. The player wins double with probability
    WIN_PROB and otherwise loses the bet.

    - State space: The player's current wealth encoded as a one-hot vector with
      size TARGET_WEALTH + 1. Note that 0 and TARGET_WEALTH are terminal states.
    - Action space: An integer in [1, TARGET_WEALTH - 1] indicating the bet amount.
    - Transition dynamics: With probability WIN_PROB, the player's wealth increases
      by the bet amount, otherwise, the player's wealth decreases by the bet amount.
      The game terminates when the player's wealth reaches TARGET_WEALTH or 0.
    - Reward: +1 if the player's wealth reaches TARGET_WEALTH, -1 if the player's
      wealth reaches 0, and 0 otherwise.
    """
    target_wealth: int
    win_prob: float
    rng: Generator

    def __init__(self, target_wealth: int, win_prob: float, seed: int):
        assert target_wealth > 0
        assert 0 <= win_prob and win_prob <= 1
        self.target_wealth = target_wealth
        self.win_prob = win_prob
        self.rng = np.random.default_rng(seed=seed)

    def reset(self) -> GamblerState:
        """
        Creates a new instance of the gambler game starting with a random initial wealth.
        Each instance is associated with a random generator with a different seed.
        """
        seed = self.rng.integers(SEED_RANGE[0], SEED_RANGE[1])
        wealth = self.rng.integers(0, self.target_wealth)
        return GamblerState(
            self.target_wealth,
            wealth,
            False,
            np.random.default_rng(seed=seed),
        )

    def step(self, state: GamblerState, action: int) -> tuple[float, Optional[GamblerState]]:
        """
        Transitions the Markov decision process at the state with the player action.
        The reward is returned, and if the episode terminates, no next state is returned.
        """
        assert not state.done
        assert action >= 1 and action <= state.wealth
        raise NotImplementedError