import numpy as np
from numpy.random import Generator
import torch

import rand

class GamblerState:
    """
    A state in the gambler Markov decision process that contains the current
    wealth and the random generator seed.
    """
    target_wealth: int
    wealth: int
    done: bool
    seed: int

    def __init__(self, target_wealth: int, wealth: int, done: bool, seed: int):
        self.target_wealth = target_wealth
        self.wealth = wealth
        self.done = done
        self.seed = seed

    def get_observation(self) -> torch.Tensor:
        """Encodes the player's current wealth as a one-hot vector."""
        observation = torch.zeros(self.target_wealth + 1, dtype=torch.float32)
        observation[self.wealth] = 1
        return observation

    def get_action_mask(self) -> torch.Tensor:
        """Encodes the legal actions given the player's current wealth."""
        action_mask = torch.zeros(self.target_wealth - 1, dtype=torch.bool)
        if not self.done:
            action_mask[:self.wealth] = True
        return action_mask

    def __repr__(self) -> str:
        """Formats the state as a string."""
        return f"GamblerState(target_wealth={self.target_wealth}, " \
            + f"wealth={self.wealth}, done={self.done}, seed={self.seed})"

class GamblerGame:
    """
    An implementation of the gambler betting game as a Markov decision process.
    The player starts with a random initial wealth and bets an integer amount at
    each time step to reach TARGET_WEALTH. The player wins double with probability
    WIN_PROB and otherwise loses the bet.

    - State space: The player's current wealth encoded as a one-hot vector with
      size TARGET_WEALTH + 1. Note that 0 and TARGET_WEALTH are terminal states.
    - Action space: An integer in the range [0, TARGET_WEALTH - 2] indicating
      1 fewer than the bet amount. Zero indexing is used for convenience.
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

    def get_state_size(self) -> int:
        """Returns the size of the 1-dimensional state vector."""
        return self.target_wealth + 1

    def get_action_size(self) -> int:
        """Returns the number of available actions."""
        return self.target_wealth - 1

    def reset(self) -> GamblerState:
        """Creates a new gambler game state starting with a random initial wealth."""
        wealth = self.rng.integers(1, self.target_wealth)
        return self.create_state(wealth)

    def create_state(self, wealth: int) -> GamblerState:
        """
        Creates a game state starting with a given wealth. Each state is initialized with
        a random generator with a different seed.
        """
        assert 1 <= wealth and wealth <= self.target_wealth - 1
        seed = rand.generate_seed(self.rng)
        return GamblerState(self.target_wealth, wealth, False, seed)

    def step(self, state: GamblerState, action: int) -> tuple[float, GamblerState]:
        """
        Advances the state in the Markov decision process with the player action. The step
        function returns the reward, and if the episode terminates, the done flag on the
        next state is marked as True.
        """
        assert not state.done
        assert 0 <= action and action <= state.wealth - 1
        bet_amount = action + 1

        rng = np.random.default_rng(seed=state.seed)
        next_wealth = None
        if rng.random() < self.win_prob:
            next_wealth = min(state.wealth + bet_amount, self.target_wealth)
        else:
            next_wealth = max(state.wealth - bet_amount, 0)
        reward = 0
        if next_wealth == 0:
            reward = 0
        elif next_wealth == self.target_wealth:
            reward = 1

        next_state = GamblerState(
            self.target_wealth,
            next_wealth,
            next_wealth == 0 or next_wealth == self.target_wealth,
            rand.generate_seed(rng),
        )
        return (reward, next_state)