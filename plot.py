import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from typing import Any

from env.environment import GamblerGame
from q_learning.deep_q import ValueNetwork

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

def get_q_network_values(model: nn.Module, target_wealth: int) -> tuple[list[int], list[int]]:
    """Queries a Q-value network for the predicted value of each state."""
    states = list(range(1, target_wealth))
    values = []
    env = GamblerGame(target_wealth, 0.5, 0)

    for wealth in states:
        with torch.no_grad():
            state = env.create_state(wealth)
            q_values = model.forward(state.get_observation())
            q_values[~state.get_action_mask()] = -np.inf
            values.append(float(torch.max(q_values).item()))

    return (states, values)

def get_q_network_policy(model: nn.Module, target_wealth: int) -> tuple[list[int], list[int]]:
    """Queries a Q-value network for the predicted bet amount at each state."""
    states = list(range(1, target_wealth))
    actions = []
    env = GamblerGame(target_wealth, 0.5, 0)

    for wealth in states:
        with torch.no_grad():
            state = env.create_state(wealth)
            q_values = model.forward(state.get_observation())
            q_values[~state.get_action_mask()] = -np.inf
            action = int(torch.argmax(q_values).item())
            actions.append(action + 1)

    return (states, actions)

def plot_training_run(data: dict, params: dict, save_path: str):
    """Plots a training curve for a training run."""
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

def plot_tabular_q(key: int, t: int):
    """Generates charts for a tabular Q-learning run."""
    data = load_data(f"data/tabular_q_data{t}_{key}.json")
    q_table = load_data(f"data/tabular_q_model{t}_{key}.json")
    env_key = get_env_key(data)
    values = get_q_table_values(q_table, data["target_wealth"])
    policy = get_q_table_policy(q_table, data["target_wealth"])

    plot_training_run(
        data,
        {
            "env_key": env_key,
            "title": "Tabular Q-learning score",
            "x_label": "Training episodes",
            "y_label": "Mean score",
            "y_limits": (0.05, 0.45),
        },
        f"charts/tabular_q_training{t}_{key}.png",
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
        f"charts/tabular_q_values{t}_{key}.png",
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
        f"charts/tabular_q_policy{t}_{key}.png",
    )

def plot_deep_q(key: int, t: int):
    """Generates charts for a deep Q-learning run."""
    data = load_data(f"data/deep_q_data{t}_{key}.json")
    model_state = torch.load(f"data/deep_q_model{t}_{key}.pt", weights_only=True)
    env_key = get_env_key(data)

    target_wealth = data["target_wealth"]
    model = ValueNetwork(target_wealth + 1, target_wealth - 1)
    model.load_state_dict(model_state)
    values = get_q_network_values(model, target_wealth)
    policy = get_q_network_policy(model, target_wealth)

    plot_training_run(
        data,
        {
            "env_key": env_key,
            "title": "Deep Q-learning score",
            "x_label": "Training episodes",
            "y_label": "Mean score",
            "y_limits": (0.05, 0.45),
        },
        f"charts/deep_q_training{t}_{key}.png",
    )
    plot_values(
        values[0],
        values[1],
        {
            "env_key": env_key,
            "title": "Deep Q-learning state values",
            "x_label": "Wealth",
            "y_label": "Value",
            "y_limits": (-0.05, 1),
        },
        f"charts/deep_q_values{t}_{key}.png",
    )
    plot_policy(
        policy[0],
        policy[1],
        {
            "env_key": env_key,
            "title": "Deep Q-learning bet amounts",
            "x_label": "Wealth",
            "y_label": "Bet amount",
            "y_limits": (0, 52),
        },
        f"charts/deep_q_policy{t}_{key}.png",
    )

# Generate charts
#for key in (99, 100):
#    plot_tabular_q(key)
plot_deep_q(99, 169850)