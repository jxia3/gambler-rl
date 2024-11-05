import numpy as np

from environment import GamblerGame
import deep_q
import rand
import tabular_q

# Game parameters
TARGET_WEALTH: int = 10
WIN_PROB: float = 0.4
SEED: int = 0
SEED_RANGE: tuple[int, int] = (0, 1_000_000_000)

# Initialize environment and train agent
rng = np.random.default_rng() # seed=SEED
env = GamblerGame(TARGET_WEALTH, WIN_PROB, rand.generate_seed(rng))
deep_q.train(env, rand.generate_seed(rng))