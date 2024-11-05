from environment import GamblerGame
import q_learning
import policy_gradient

import numpy as np

# Game parameters
TARGET_WEALTH: int = 20
WIN_PROB: float = 0.5
SEED: int = 0
SEED_RANGE = (0, 1_000_000_000)

# Initialize environment and train agent
rng = np.random.default_rng(seed=SEED)
env = GamblerGame(TARGET_WEALTH, WIN_PROB, rng.integers(SEED_RANGE[0], SEED_RANGE[1]))
q_learning.train(env, rng.integers(SEED_RANGE[0], SEED_RANGE[1]))