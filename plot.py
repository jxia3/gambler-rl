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

def plot_value_policy(data_path: str, model_path: str, params: dict, save_path: str):
    """Plots the predicted bet amount for each state for a Q-table."""
    data = load_data(data_path)
    q_table = load_data(model_path)
    env_key = get_env_key(data)

    x_values = list(range(1, data["target_wealth"]))
    y_values = []
    for wealth in x_values:
        action = 0
        for a in range(1, wealth):
            if q_table[wealth][a] > q_table[wealth][action]:
                action = a
        y_values.append(action + 1)

    figure, axes = plt.subplots()
    axes.scatter(x_values, y_values)
    axes.set_ylim(params["y_limits"][0], params["y_limits"][1])

    axes.set_title(f"{env_key} {params['title']}")
    axes.set_xlabel(params["x_label"])
    axes.set_ylabel(params["y_label"])

    figure.savefig(save_path, bbox_inches="tight")

'''
for run in (99, 100):
    data_path = f"data/tabular_q_data_{run}.json"
    model_path = f"data/tabular_q_model_{run}.json"
    plot_run(
        data_path,
        {
            "title": "Tabular Q-learning score",
            "x_label": "Training episodes",
            "y_label": "Mean score",
            "y_limits": (0.05, 0.45),
        },
        f"charts/tabular_q_training_{run}.png",
    )
    plot_state_values(
        data_path,
        model_path,
        {
            "title": "Tabular Q-learning state values",
            "x_label": "Wealth",
            "y_label": "Value",
            "y_limits": (-0.05, 1),
        },
        f"charts/tabular_q_values_{run}.png",
    )
    plot_value_policy(
        data_path,
        model_path,
        {
            "title": "Tabular Q-learning bet amounts",
            "x_label": "Wealth",
            "y_label": "Bet amount",
            "y_limits": (0, 52),
        },
        f"charts/tabular_q_policy_{run}.png",
    )
'''

plot_run(
    "data/deep_q_data3_99.json",
    {
        "title": "Tabular Q-learning score",
        "x_label": "Training episodes",
        "y_label": "Mean score",
        "y_limits": (0.05, 0.45),
    },
    "charts/deep_q_training3_99.png",
)