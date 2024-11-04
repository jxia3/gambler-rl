import numpy as np
from numpy.random import Generator
from typing import Optional

class GamblerState:
    """
    A state in the gambler Markov decision process that contains the current
    wealth and the random generator for the episode.
    """
    target_wealth: int
    wealth: int
    rng: Generator

    def __init__(self, target_wealth: int, wealth: int, rng: Generator):
        self.target_wealth = target_wealth
        self.wealth = wealth
        self.rng = rng

    def get_observation(self) -> np.ndarray:
        """Encodes the player's current wealth as a one-hot vector."""
        observation = np.zeros(self.target_wealth - 1, dtype=np.int32)
        observation[self.wealth - 1] = 1
        return observation

    def get_action_mask(self) -> np.ndarray:
        """Encodes the legal actions given the player's current wealth."""
        action_mask = np.zeros(self.target_wealth - 1, dtype=np.int32)
        action_mask[:self.wealth] = 1
        return action_mask

class GamblerGame:
    """
    An implementation of the gambler betting game as a Markov decision process.
    The player starts with a random initial wealth and bets an integer amount at
    each time step to reach TARGET_WEALTH. The player wins double with probability
    WIN_PROB and otherwise loses the bet.

    - State space: The player's current wealth encoded as a one-hot vector with
      size TARGET_WEALTH - 1.
    - Action space: A one-hot vector with size TARGET_WEALTH - 1 indicating the
      amount to bet. Illegal actions are masked.
    - Transition dynamics: With probability WIN_PROB, the player's wealth increases
      by the bet amount, otherwise, the player's wealth decreases by the bet amount.
      The game terminates when the player's wealth reaches TARGET_WEALTH or 0.
    - Reward: +1 if the player's wealth reaches TARGET_WEALTH, -1 if the player's
      wealth reaches 0, and 0 otherwise.
    """
    TARGET_WEALTH: int = 20
    WIN_PROB: float = 0.5

    seed: int

    def __init__(self, seed: int = 0):
        self.seed = seed

    def reset(self) -> tuple[GamblerState]:
        """
        Creates a new instance of the gambler game starting with a random initial wealth.
        Each instance is associated with a random generator with a different seed.
        """
        raise NotImplementedError

    def step(self, action: np.ndarray) -> tuple[float, Optional[GamblerState]]:
        """
        Transitions the Markov decision process at the state with the player action.
        The reward is returned, and if the episode terminates, no next state is returned.
        """
        raise NotImplementedError