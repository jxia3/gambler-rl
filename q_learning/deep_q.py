import numpy as np
from numpy.random import Generator
import torch
import torch.nn as nn
import torch.nn.functional as F

from env.environment import GamblerGame
from env.evaluation import Evaluation
from q_learning.buffer import TensorTransitionBuffer, Transition
import rand

# Training parameters
HIDDEN_SIZE: int = 20
DISCOUNT_RATE: float = 1
LEARNING_RATE: float = 0.01
SYNC_INTERVAL: int = 10

INITIAL_EXPLORE: float = 1
EXPLORE_DECAY: float = 0.99995
MIN_EXPLORE: float = 0.01
BUFFER_SIZE: int = 100_000
BATCH_SIZE: int = 800

EPISODES: int = 500_000
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

def train(env: GamblerGame, evaluation: Evaluation, seed: int) -> tuple[nn.Module, dict]:
    """Trains a deep Q-learning agent on the gambler Markov decision process."""
    rng = rand.create_generator(seed)
    torch.manual_seed(rand.generate_seed(rng))

    # Actions are sampled from the policy network and value targets are computed
    # using the target network. Keeping the target network fixed for several
    # training iterations at a time stabilizes the training.
    policy_network = ValueNetwork(env.get_state_size(), env.get_action_size())
    target_network = ValueNetwork(env.get_state_size(), env.get_action_size())
    target_network.load_state_dict(policy_network.state_dict())

    # Initialize training
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=LEARNING_RATE)
    transitions = TensorTransitionBuffer(
        BUFFER_SIZE,
        env.get_state_size(),
        env.get_action_size(),
        rng,
    )
    explore_factor = INITIAL_EXPLORE
    scores = {}
    scores[0] = evaluation.evaluate_q_network(policy_network)

    for episode in range(1, EPISODES + 1):
        # Simulate trajectory with the current policy network
        trajectory = run_rollout(env, policy_network, explore_factor, rng)
        transitions.insert(trajectory)
        if len(transitions) < BATCH_SIZE:
            continue

        # Sample random batch for training and stack the transition data tensors
        # for efficient batch neural network queries

        train_sample = transitions.sample(BATCH_SIZE)
        states = train_sample.observations
        actions = train_sample.actions
        rewards = train_sample.rewards
        next_states = train_sample.next_observations
        next_action_masks = train_sample.next_action_masks
        done_mask = train_sample.done_mask

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
            score = evaluation.evaluate_q_network(policy_network)
            scores[episode] = score
            print(f"[{episode}] score={round(score, 4)}, loss={round(float(loss.item()), 4)}, "
                + f"explore={round(explore_factor, 4)}")
            with torch.no_grad():
                state = env.create_state(94)
                values = policy_network.forward(state.get_observation())
                print("values:", values.numpy())

    return (policy_network, scores)

def save_model(model: nn.Module, save_path: str):
    """Saves the model weights in a file in a readable JSON format."""