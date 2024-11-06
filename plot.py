import json
import matplotlib.pyplot as plt
import torch
from typing import Any

def load_data(data_path: str) -> Any:
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

def get_q_table_values(q_table: list[list[int]], target_wealth: int) -> tuple[list[int], list[int]]:
    """Computes the predicted value of each state for a Q-table."""
    states = list(range(1, target_wealth))
    values = []
    for wealth in states:
        q_values = q_table[wealth][:wealth]
        values.append(max(q_values))
    return (states, values)

def get_q_table_policy(q_table: list[list[int]], target_wealth: int) -> tuple[list[int], list[int]]:
    """Computes the predicted bet amount at each state for a Q-table."""
    states = list(range(1, target_wealth))
    actions = []
    for wealth in states:
        action = 0
        for a in range(1, wealth):
            if q_table[wealth][a] > q_table[wealth][action]:
                action = a
        actions.append(action + 1)
    return (states, actions)

def plot_training_run(scores: dict, params: dict, save_path: str):
    """Plots a training curve for a training run."""
    x_values = []
    y_values = []
    for score in scores:
        x_values.append(int(score))
        y_values.append(float(scores[score]))

    figure, axes = plt.subplots()
    axes.plot(x_values, y_values)
    axes.axhline(data["optimal_score"], linestyle="dashed", c="green")
    axes.axhline(data["random_score"], linestyle="dashed", c="red")
    axes.set_ylim(params["y_limits"][0], params["y_limits"][1])

    axes.set_title(f"{params['env_key']} {params['title']}")
    axes.set_xlabel(params["x_label"])
    axes.set_ylabel(params["y_label"])

    figure.savefig(save_path, bbox_inches="tight")

def plot_values(states: list[int], values: list[int], params: dict, save_path: str):
    """Plots the predicted value of each state."""
    figure, axes = plt.subplots()
    axes.plot(states, values)
    axes.set_ylim(params["y_limits"][0], params["y_limits"][1])

    axes.set_title(f"{params['env_key']} {params['title']}")
    axes.set_xlabel(params["x_label"])
    axes.set_ylabel(params["y_label"])

    figure.savefig(save_path, bbox_inches="tight")

def plot_policy(states: list[int], actions: list[int], params: dict, save_path: str):
    """Plots the predicted bet amount for each state for a Q-table."""
    figure, axes = plt.subplots()
    axes.scatter(states, actions)
    axes.set_ylim(params["y_limits"][0], params["y_limits"][1])

    axes.set_title(f"{params['env_key']} {params['title']}")
    axes.set_xlabel(params["x_label"])
    axes.set_ylabel(params["y_label"])

    figure.savefig(save_path, bbox_inches="tight")

for run in (99, 100):
    data = load_data(f"data/tabular_q_data_{run}.json")
    q_table = load_data(f"data/tabular_q_model_{run}.json")
    env_key = get_env_key(data)
    values = get_q_table_values(q_table, data["target_wealth"])
    policy = get_q_table_policy(q_table, data["target_wealth"])

    plot_training_run(
        data["scores"],
        {
            "env_key": env_key,
            "title": "Tabular Q-learning score",
            "x_label": "Training episodes",
            "y_label": "Mean score",
            "y_limits": (0.05, 0.45),
        },
        f"charts/tabular_q_training_{run}.png",
    )
    plot_values(
        values[0],
        values[1],
        {
            "env_key": env_key,
            "title": "Tabular Q-learning state values",
            "x_label": "Wealth",
            "y_label": "Value",
            "y_limits": (-0.05, 1),
        },
        f"charts/tabular_q_values_{run}.png",
    )
    plot_policy(
        policy[0],
        policy[1],
        {
            "env_key": env_key,
            "title": "Tabular Q-learning bet amounts",
            "x_label": "Wealth",
            "y_label": "Bet amount",
            "y_limits": (0, 52),
        },
        f"charts/tabular_q_policy_{run}.png",
    )

'''
plot_run(
    "data/deep_q_data_99.json",
    {
        "title": "Tabular Q-learning score",
        "x_label": "Training episodes",
        "y_label": "Mean score",
        "y_limits": (0.05, 0.45),
    },
    "charts/deep_q_training_99.png",
)
plot_q_network_values(
    "data/deep_q_data_99.json",
    "data/deep_q_model_99.pt",
    {
        "title": "Tabular Q-learning score",
        "x_label": "Training episodes",
        "y_label": "Mean score",
        "y_limits": (0.05, 0.45),
    },
    "charts/deep_q_values_99.png",
)
'''