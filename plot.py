import json
import matplotlib.pyplot as plt

def load_data(data_path: str) -> dict:
    """Loads training data from a JSON file."""
    data = None
    with open(data_path, "r") as file:
        data = json.load(file)
    assert data is not None
    return data

def get_env_key(data: dict) -> str:
    """Formats the gambler environment parameters as a string key."""
    target_wealth = data["target_wealth"]
    win_prob = data["win_prob"]
    return f"[Gambler-{target_wealth}-{win_prob}]"

def plot_run(data_path: str, params: dict, save_path: str):
    """Plots a training curve for a training run."""
    data = load_data(data_path)
    env_key = get_env_key(data)

    x_values = []
    y_values = []
    for score in data["scores"]:
        x_values.append(int(score))
        y_values.append(float(data["scores"][score]))

    figure, axes = plt.subplots()
    axes.plot(x_values, y_values)
    axes.axhline(data["optimal_score"], linestyle="dashed", c="green")
    axes.axhline(data["random_score"], linestyle="dashed", c="red")
    axes.set_ylim(params["y_limits"][0], params["y_limits"][1])

    axes.set_title(f"{env_key} {params['title']}")
    axes.set_xlabel(params["x_label"])
    axes.set_ylabel(params["y_label"])

    figure.savefig(save_path, bbox_inches="tight")

def plot_state_values(data_path: str, model_path: str, params: dict, save_path: str):
    """Plots the predicted value of each state for a Q-table."""
    data = load_data(data_path)
    q_table = load_data(model_path)
    env_key = get_env_key(data)

    x_values = list(range(1, data["target_wealth"]))
    y_values = []
    for wealth in x_values:
        values = q_table[wealth][:wealth]
        y_values.append(max(values))

    figure, axes = plt.subplots()
    axes.plot(x_values, y_values)
    axes.set_ylim(params["y_limits"][0], params["y_limits"][1])

    axes.set_title(f"{env_key} {params['title']}")
    axes.set_xlabel(params["x_label"])
    axes.set_ylabel(params["y_label"])

    figure.savefig(save_path, bbox_inches="tight")

plot_run(
    "data/tabular_q_data_99.json",
    {
        "title": "Tabular Q-learning score",
        "x_label": "Training episodes",
        "y_label": "Mean score",
        "y_limits": (0.05, 0.45),
    },
    "charts/tabular_q_training_99.png",
)
plot_state_values(
    "data/tabular_q_data_99.json",
    "data/tabular_q_model_99.json",
    {
        "title": "Tabular Q-learning state values",
        "x_label": "Wealth",
        "y_label": "Value",
        "y_limits": (-0.05, 1),
    },
    "charts/tabular_q_values_99.png"
)