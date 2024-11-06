import json
from typing import Any

from env.environment import GamblerGame
from env.evaluation import Evaluation
from q_learning import deep_q, tabular_q
import rand

# Game parameters
TARGET_WEALTH: int = 99
WIN_PROB: float = 0.4
EVAL_EPISODES: int = 10_000
BASELINE_EPISODES: int = 100_000
SEED: int = 1

# Training configuration
TRAIN_CONFIG: dict[str, Any] = {
    "tabular_q": {
        "train_fn": tabular_q.train,
        "save_fn": tabular_q.save_model,
        "data_path": f"data/tabular_q_data_{TARGET_WEALTH}.json",
        "model_path": f"data/tabular_q_model_{TARGET_WEALTH}.json",
    },
    "deep_q": {
        "train_fn": deep_q.train,
        "save_fn": deep_q.save_model,
        "data_path": f"data/deep_q_data99002_{TARGET_WEALTH}.json",
        "model_path": f"data/deep_q_model99002_{TARGET_WEALTH}.pt",
    },
}
MODEL = "deep_q"

# Initialize environment and evaluation context
rng = rand.create_generator(SEED)
train_env = GamblerGame(TARGET_WEALTH, WIN_PROB, rand.generate_seed(rng))
eval_env = GamblerGame(TARGET_WEALTH, WIN_PROB, rand.generate_seed(rng))
evaluation = Evaluation(eval_env, EVAL_EPISODES, rand.generate_seed(rng))

# Train agent and save results
config = TRAIN_CONFIG[MODEL]
model, scores = config["train_fn"](train_env, evaluation, rand.generate_seed(rng))
optimal_score = evaluation.evaluate_optimal(episodes=BASELINE_EPISODES)
random_score = evaluation.evaluate_random(episodes=BASELINE_EPISODES)
print(f"Optimal score: {round(optimal_score, 4)}")
print(f"Random score: {round(random_score, 4)}")

with open(config["data_path"], "w") as file:
    file.write(json.dumps({
        "target_wealth": TARGET_WEALTH,
        "win_prob": WIN_PROB,
        "optimal_score": optimal_score,
        "random_score": random_score,
        "scores": scores,
    }, indent=4))
print(f"Saved data to {config['data_path']}")

config["save_fn"](model, config["model_path"])
print(f"Saved model to {config['model_path']}")