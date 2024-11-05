import json
import numpy as np
from typing import Any

from environment import GamblerGame
from eval.evaluation import Evaluation
from q_learning import deep_q, tabular_q
import rand

# Game parameters
TARGET_WEALTH: int = 99
WIN_PROB: float = 0.4
EVAL_EPISODES: int = 10_000
OPTIMAL_EPISODES: int = 100_000
SEED: int = 0

# Training configuration
TRAIN_CONFIG: dict[str, Any] = {
    "tabular_q": {
        "train_fn": tabular_q.train,
        "save_path": "data/tabular_q.json",
    },
    "deep_q": {
        "train_fn": deep_q.train,
        "save_path": "data/deep_q.json",
    },
}
MODEL = "tabular_q"

# Initialize environment and evaluation context
rng = np.random.default_rng(seed=SEED)
train_env = GamblerGame(TARGET_WEALTH, WIN_PROB, rand.generate_seed(rng))
eval_env = GamblerGame(TARGET_WEALTH, WIN_PROB, rand.generate_seed(rng))
evaluation = Evaluation(eval_env, EVAL_EPISODES, rand.generate_seed(rng))

# Train agent and save results
config = TRAIN_CONFIG[MODEL]
scores = config["train_fn"](train_env, evaluation, rand.generate_seed(rng))
optimal_score = evaluation.evaluate_optimal(episodes=OPTIMAL_EPISODES)
print(f"Optimal score: {round(optimal_score, 4)}")

with open(config["save_path"], "w") as file:
    file.write(json.dumps({
        "target_wealth": TARGET_WEALTH,
        "win_prob": WIN_PROB,
        "optimal_score": optimal_score,
        "scores": scores,
    }))
print(f"Saved data to {config['save_path']}")