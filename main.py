import numpy as np

from environment import GamblerGame
import deep_q
import tabular_q

# Game parameters
TARGET_WEALTH: int = 10
WIN_PROB: float = 0.6
SEED: int = 0
SEED_RANGE: tuple[int, int] = (0, 1_000_000_000)

# Initialize environment and train agent
rng = np.random.default_rng(seed=SEED)
env = GamblerGame(TARGET_WEALTH, WIN_PROB, rng.integers(SEED_RANGE[0], SEED_RANGE[1]))
tabular_q.train(env, rng.integers(SEED_RANGE[0], SEED_RANGE[1]))