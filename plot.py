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
    axes.axhline(data["optimal_score"], linestyle="dashed")
    axes.set_title(f"{env_key} {params['title']}")
    axes.set_xlabel(params["x_label"])
    axes.set_ylabel(params["y_label"])

    figure.savefig(save_path, bbox_inches="tight")

plot_run(
    "data/tabular_q_data4.json",
    {
        "title": "Tabular Q-learning score",
        "x_label": "Training episodes",
        "y_label": "Mean score",
    },
    "charts/tabular_q4.png",
)