import numpy as np

from environment import GamblerGame
from q_learning import deep_q, tabular_q
import rand

# Game parameters
TARGET_WEALTH: int = 10
WIN_PROB: float = 0.6
SEED: int = 0

# Initialize environment and train agent
rng = np.random.default_rng() # seed=SEED
env = GamblerGame(TARGET_WEALTH, WIN_PROB, rand.generate_seed(rng))
tabular_q.train(env, rand.generate_seed(rng))