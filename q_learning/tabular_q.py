import json
import numpy as np
from numpy.random import Generator

from env.environment import GamblerGame
from env.evaluation import Evaluation
from q_learning.buffer import Transition, TransitionBuffer
import rand

# Training parameters
INIT_SDEV: float = 0.01
DISCOUNT_RATE: float = 0.99
INITIAL_LEARNING_RATE: float = 0.02
LEARNING_RATE_DECAY: float = 0.99999
MIN_LEARNING_RATE: float = 0.0005

INITIAL_EXPLORE: float = 1
EXPLORE_DECAY: float = 0.99995
MIN_EXPLORE: float = 0.01
BUFFER_SIZE: int = 100_000
BATCH_SIZE: int = 800

EPISODES: int = 500_000
CLIP_END: int = 20_000
MAX_VALUE: float = 100
LOG_INTERVAL: int = 1000

def run_rollout(
    env: GamblerGame,
    q_table: np.ndarray,
    explore_factor: float,
    rng: Generator,
) -> list[Transition]:
    """
    Simulates a trajectory in the Markov decision process with the explicit Q-table.
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
            values = q_table[state.get_index()].copy()
            values[~state.get_action_mask().numpy()] = -np.inf
            action = int(np.argmax(values))

        reward, next_state = env.step(state, action)
        transitions.append(Transition(state, action, reward, next_state))
        state = next_state

    return transitions

def train(env: GamblerGame, evaluation: Evaluation, seed: int) -> tuple[np.ndarray, dict]:
    """Trains a tabular Q-learning agent on the gambler Markov decision process."""
    rng = rand.create_generator(seed)

    # Initialize Q-table with random values from a normal distribution
    q_table = np.zeros((env.get_state_size(), env.get_action_size()), dtype=np.float32)
    for s in range(env.get_state_size()):
        for a in range(env.get_action_size()):
            q_table[s][a] = rng.normal(loc=0, scale=INIT_SDEV)

    # Initialize training
    transitions = TransitionBuffer(BUFFER_SIZE, rng)
    learning_rate = INITIAL_LEARNING_RATE
    explore_factor = INITIAL_EXPLORE
    scores = {}
    scores[0] = evaluation.evaluate_q_table(q_table)

    for episode in range(1, EPISODES + 1):
        # Simulate trajectory with the current Q-table
        trajectory = run_rollout(env, q_table, explore_factor, rng)
        transitions.insert(trajectory)
        if len(transitions) < BATCH_SIZE:
            continue

        # Sample random batch for training
        train_sample = transitions.sample(BATCH_SIZE)
        rewards = np.array([t.reward for t in train_sample], dtype=np.float32)
        next_indices = np.array([t.next_state.get_index() for t in train_sample], dtype=np.int32)
        done_mask = np.array([t.next_state.done for t in train_sample], dtype=np.bool_)

        # Update the Q-values using the discounted dynamic programming equation
        prev_table = q_table.copy()
        q_max = prev_table.max(axis=1)
        targets = rewards + DISCOUNT_RATE * q_max[next_indices] * (~done_mask)
        for t in range(len(train_sample)):
            state_index = train_sample[t].state.get_index()
            q_table[state_index][train_sample[t].action] += \
                learning_rate * (targets[t] - prev_table[state_index][train_sample[t].action])

        # Clip large values in the Q-table at the beginning of training
        if episode < CLIP_END:
            q_table = q_table.clip(-MAX_VALUE, MAX_VALUE)

        # Decay the explore factor and learning rate
        if explore_factor > MIN_EXPLORE:
            explore_factor = max(explore_factor * EXPLORE_DECAY, MIN_EXPLORE)
        if episode >= CLIP_END and learning_rate > MIN_LEARNING_RATE:
            learning_rate = max(learning_rate * LEARNING_RATE_DECAY, MIN_LEARNING_RATE)

        # Log statistics
        if episode % LOG_INTERVAL == 0:
            score = evaluation.evaluate_q_table(q_table)
            scores[episode] = score
            print(f"[{episode}] score={round(score, 4)}, lr={round(learning_rate, 4)}, "
                + f"explore={round(explore_factor, 4)}")

    return (q_table, scores)

def save_model(q_table: np.ndarray, save_path: str):
    """Saves the Q-table in a file in a readable JSON format."""
    table = []
    for row in q_table:
        table.append([float(v) for v in row])
    with open(save_path, "w") as file:
        file.write(json.dumps(table, indent=4))