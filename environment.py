import numpy

class GamblerState:
    """
    An internal environment state keeping track of the current wealth and
    random generator.
    """
    wealth: int
    seed: int

class GamblerGame:
    """
    An implementation of the gambler betting game as a Markov decision process.
    The player starts with a random initial wealth and bets an integer amount at
    each time step to reach TARGET_WEALTH. The player wins double with probability
    WIN_PROB and otherwise loses the bet.

    - State space: The player's current wealth encoded as a one-hot vector with
      size TARGET_WEALTH - 1.
    - Action space: A one-hot vector with size TARGET_WEALTH - 1 indicating the
      amount to bet.
    - Transition dynamics: With probability WIN_PROB, the player's wealth increases
      by the bet amount, otherwise, the player's wealth decreases by the bet amount.
      The game terminates when the player's wealth reaches TARGET_WEALTH or 0.
    - Reward: +1 if the player's wealth reaches TARGET_WEALTH, -1 if the player's
      wealth reaches 0, and 0 otherwise.
    """
    TARGET_WEALTH: int = 20
    WIN_PROB: float = 0.5