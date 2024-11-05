import numpy as np
from numpy.random import Generator
import torch
import torch.nn as nn
import torch.nn.functional as F

from q_learning.buffer import Transition, TransitionBuffer
from environment import GamblerGame, GamblerState
import rand

# Training parameters
HIDDEN_SIZE: int = 32
DISCOUNT_RATE: float = 1
LEARNING_RATE: float = 0.005
SYNC_INTERVAL: int = 10

INITIAL_EXPLORE: float = 1
EXPLORE_DECAY: float = 0.999
MIN_EXPLORE: float = 0.02
BUFFER_SIZE: int = 2000
BATCH_SIZE: int = 100

EPISODES: int = 100000
LOG_INTERVAL: int = 100

class ValueNetwork(nn.Module):
    """
    A Q-value neural network with 1 hidden layer that predicts the discounted value
    for each action at a state.
    """
    def __init__(self, state_size: int, action_size: int):
        super(ValueNetwork, self).__init__()
        self.hidden = nn.Linear(state_size, HIDDEN_SIZE)
        self.output = nn.Linear(HIDDEN_SIZE, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.hidden(state)
        out = F.relu(out)
        out = self.output(out)
        return out

def run_rollout(
    env: GamblerGame,
    model: ValueNetwork,
    explore_factor: float,
    rng: Generator,
) -> list[Transition]:
    """
    Simulates a trajectory in the Markov decision process with the Q-value network.
    Random 'explore' actions are chosen with probability explore_factor and otherwise
    'exploit' actions with the maximum predicted value are chosen.
    """
    transitions = []
    state = env.reset()

    while not state.done:
        action = None
        if rng.random() < explore_factor:
            # Take random 'explore' action
            actions = np.nonzero(state.get_action_mask().numpy())[0]
            action = rng.choice(actions)
        else:
            # Take 'exploit' action
            with torch.no_grad():
                values = model.forward(state.get_observation())
                values[~state.get_action_mask()] = -np.inf
                action = int(torch.argmax(values).item())

        reward, next_state = env.step(state, action)
        transitions.append(Transition(state, action, reward, next_state))
        state = next_state

    return transitions

def get_q_table(env: GamblerGame, model: ValueNetwork) -> np.ndarray:
    """
    Queries the model at each state to calculate the Q-table. For environments with large
    state spaces or continuous states, storing the Q-table explicitly is intractable.
    The gambler game has a small state space so we can represent the Q-table as a
    2-dimensional array. Note that we only use the Q-table for efficient model evaluation.
    """
    q_table = np.zeros((env.get_state_size(), env.get_action_size()))
    for state_index in range(1, env.get_state_size() - 1):
        state = env.create_state(state_index)
        print(state)

    return np.array([])

def train(env: GamblerGame, seed: int):
    """Trains a deep Q-learning agent on the gambler Markov decision process."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(rand.generate_seed(rng))

    # Actions are sampled from the policy network and value targets are computed
    # using the target network. Keeping the target network fixed for several
    # training iterations at a time stabilizes the training.
    policy_network = ValueNetwork(env.get_state_size(), env.get_action_size())
    target_network = ValueNetwork(env.get_state_size(), env.get_action_size())
    target_network.load_state_dict(policy_network.state_dict())

    # Initialize training
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=LEARNING_RATE)
    transitions = TransitionBuffer(BUFFER_SIZE, rng)
    explore_factor = INITIAL_EXPLORE

    for episode in range(1, EPISODES + 1):
        # Simulate trajectory with the current policy network
        trajectory = run_rollout(env, policy_network, explore_factor, rng)
        transitions.insert(trajectory)
        if len(transitions) < BATCH_SIZE:
            continue

        # Sample random batch for training and stack the transition data tensors
        # for efficient batch neural network queries
        train_sample = transitions.sample(BATCH_SIZE)
        states = torch.vstack([t.state.get_observation() for t in train_sample])
        actions = torch.vstack([torch.tensor(t.action, dtype=torch.int64) for t in train_sample])
        rewards = torch.tensor([t.reward for t in train_sample], dtype=torch.float32)
        next_states = torch.vstack([t.next_state.get_observation() for t in train_sample])
        next_action_masks = torch.vstack([t.next_state.get_action_mask() for t in train_sample])
        done_mask = torch.tensor([t.next_state.done for t in train_sample], dtype=torch.bool)

        # Get the current value prediction and compute the value targets
        # using the discounted dynamic programming equation
        predicted = policy_network.forward(states).gather(1, actions).flatten()
        targets = rewards
        with torch.no_grad():
            next_values = target_network.forward(next_states)
            next_values[~next_action_masks] = -np.inf
            next_values = next_values.max(1).values
            next_values[done_mask] = 0
            targets += DISCOUNT_RATE * next_values

        # Perform gradient descent with respect to the mean squared error loss
        optimizer.zero_grad()
        loss = nn.MSELoss()(predicted, targets)
        loss.backward()
        optimizer.step()

        # Decay the explore factor
        if explore_factor > MIN_EXPLORE:
            explore_factor = max(explore_factor * EXPLORE_DECAY, MIN_EXPLORE)

        # Sync the policy network with the target network
        if episode % SYNC_INTERVAL == 0:
            target_network.load_state_dict(policy_network.state_dict())

        # Log statistics
        if episode % LOG_INTERVAL == 0:
            print(f"[{episode}] Loss: {loss.item()}")
            print(f"Explore factor: {explore_factor}")
            print()