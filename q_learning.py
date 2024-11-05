import numpy as np
from numpy.random import Generator
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import GamblerGame, GamblerState

# The inclusive integer range for sampling PyTorch seeds
SEED_RANGE: tuple[int, int] = (0, 1_000_000_000)

# Training parameters
STATE_SIZE: int = 21
ACTION_SIZE: int = STATE_SIZE - 2
HIDDEN_SIZE: int = 64
DISCOUNT_RATE: float = 0.97
LEARNING_RATE: float = 0.001
SYNC_INTERVAL: int = 4

INITIAL_EXPLORE: float = 1
EXPLORE_DECAY: float = 0.9995
MIN_EXPLORE: float = 0.1
BUFFER_SIZE: int = 5000
BATCH_SIZE: int = 256

EPISODES: int = 10000
LOG_INTERVAL: int = 100

class ValueNetwork(nn.Module):
    """
    A Q-value neural network with 1 hidden layer that predicts the discounted
    value for each action at a state.
    """
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.hidden = nn.Linear(STATE_SIZE, HIDDEN_SIZE)
        self.output = nn.Linear(HIDDEN_SIZE, ACTION_SIZE)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.hidden(state)
        out = F.relu(out)
        out = self.output(out)
        return out

class Transition:
    """
    A tuple (s_t, a_t, r_t, s_{t + 1}) tracking a state transition in the Markov
    decision process. Transitions are stored in a buffer used for training.
    """
    state: GamblerState
    action: int
    reward: float
    next_state: GamblerState

    def __init__(self, state: GamblerState, action: int, reward: float, next_state: GamblerState):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    def __repr__(self) -> str:
        return f"({self.state}, {self.action}, {self.reward}, {self.next_state})"

class TransitionBuffer:
    """A buffer that stores previous transitions and supports sampling for training."""
    transitions: list[Transition]
    index: int
    rng: Generator

    def __init__(self, rng: Generator):
        self.transitions = []
        self.index = 0
        self.rng = rng

    def insert(self, transitions: list[Transition]):
        """Adds transitions to the buffer."""
        for transition in transitions:
            if len(self.transitions) < BUFFER_SIZE:
                self.transitions.append(transition)
            else:
                self.transitions[self.index] = transition
                self.index = (self.index + 1) % len(self.transitions)

    def sample(self, count: int) -> list[Transition]:
        """Returns random transitions from the buffer."""
        assert 1 <= count and count <= len(self.transitions)
        indices = self.rng.choice(len(self.transitions), size=count, replace=False)
        sample = []
        for index in indices:
            sample.append(self.transitions[index])
        return sample

    def __len__(self) -> int:
        """Returns the number of transitions in the buffer."""
        return len(self.transitions)

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

def get_model_policy(model: ValueNetwork) -> tuple[list[int], list[int]]:
    """Queries the model at each state to calculate the predicted policy."""
    states = []
    actions = []

    for state in range(1, STATE_SIZE - 1):
        observation = torch.zeros(STATE_SIZE, dtype=torch.float32)
        observation[state] = 1
        action_mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
        action_mask[:state] = True

        states.append(state)
        with torch.no_grad():
            values = model.forward(observation)
            values[~action_mask] = -np.inf
            actions.append(int(torch.argmax(values).item()))

    return (states, actions)

def train(env: GamblerGame, seed: int):
    """Trains a Deep Q-Learning agent on the gambler Markov decision process."""
    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(rng.integers(SEED_RANGE[0], SEED_RANGE[1]))

    # Actions are sampled from the policy network and value targets are computed
    # using the target network. Keeping the target network fixed for several
    # training iterations at a time stabilizes the training.
    policy_network = ValueNetwork()
    target_network = ValueNetwork()
    target_network.load_state_dict(policy_network.state_dict())

    # Initialize training
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=LEARNING_RATE)
    transitions = TransitionBuffer(rng)
    explore_factor = INITIAL_EXPLORE
    policies = { 0: get_model_policy(policy_network) }

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
        done_mask = torch.tensor([t.next_state.done for t in train_sample], dtype=torch.bool)

        # Get the current value prediction and compute the value targets
        # using the discounted dynamic programming equation
        predicted = policy_network.forward(states).gather(1, actions).flatten()
        targets = rewards
        with torch.no_grad():
            next_values = target_network.forward(next_states).max(1).values
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
            policy = get_model_policy(policy_network)
            policies[episode] = policy
            print(f"[{episode}] Loss: {loss.item()}")
            print(f"Explore factor: {explore_factor}")
            print(policy[0])
            print(policy[1])
            print()