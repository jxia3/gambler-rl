import numpy as np

from environment import GamblerGame
from eval.evaluation import Evaluation
from q_learning import deep_q, tabular_q
import rand

# Game parameters
TARGET_WEALTH: int = 20
WIN_PROB: float = 0.4
EVAL_EPISODES: int = 5000
SEED: int = 0

# Initialize environment and train agent
rng = np.random.default_rng() # seed=SEED
train_env = GamblerGame(TARGET_WEALTH, WIN_PROB, rand.generate_seed(rng))
eval_env = GamblerGame(TARGET_WEALTH, WIN_PROB, rand.generate_seed(rng))
evaluation = Evaluation(eval_env, EVAL_EPISODES, rand.generate_seed(rng))
tabular_q.train(train_env, evaluation, rand.generate_seed(rng))