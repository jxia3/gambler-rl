import numpy as np
from numpy.random import Generator

from environment import GamblerGame
from eval.evaluation import Evaluation
from q_learning.buffer import Transition, TransitionBuffer

# Q-table evaluation parameters
EVAL_EPISODES: int = 10000
EVAL_SEED: int = 1000

# Training parameters
INIT_SDEV: float = 0.1
DISCOUNT_RATE: float = 1
INITIAL_LEARNING_RATE: float = 0.02
LEARNING_RATE_DECAY: float = 0.9999
MIN_LEARNING_RATE: float = 0.001

INITIAL_EXPLORE: float = 1
EXPLORE_DECAY: float = 0.9992
MIN_EXPLORE: float = 0.01
BUFFER_SIZE: int = 10000
BATCH_SIZE: int = 100

EPISODES: int = 30000
LOG_INTERVAL: int = 500

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

def train(env: GamblerGame, evaluation: Evaluation, seed: int):
    """Trains a tabular Q-learning agent on the gambler Markov decision process."""
    rng = np.random.default_rng(seed)

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
        q_max = q_table.max(axis=1)
        targets = rewards + DISCOUNT_RATE * q_max[next_indices] * (~done_mask)
        for t in range(len(train_sample)):
            state_index = train_sample[t].state.get_index()
            q_table[state_index][train_sample[t].action] += \
                learning_rate * (targets[t] - q_table[state_index][train_sample[t].action])

        # Decay the explore factor and learning rate
        if explore_factor > MIN_EXPLORE:
            explore_factor = max(explore_factor * EXPLORE_DECAY, MIN_EXPLORE)
        if learning_rate > MIN_LEARNING_RATE:
            learning_rate = max(learning_rate * LEARNING_RATE_DECAY, MIN_LEARNING_RATE)

        # Log statistics
        if episode % LOG_INTERVAL == 0:
            score = evaluation.evaluate_q_table(q_table)
            scores[episode] = score
            print(f"[{episode}] score={round(score, 4)}, lr={round(learning_rate, 4)}, "
                + f"explore={round(explore_factor, 4)}")

    print(scores)