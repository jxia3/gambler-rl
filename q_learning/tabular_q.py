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
LEARNING_RATE_DECAY: float = 0.9998
MIN_LEARNING_RATE: float = 0.001

INITIAL_EXPLORE: float = 1
EXPLORE_DECAY: float = 0.9994
MIN_EXPLORE: float = 0.02
BUFFER_SIZE: int = 2000
BATCH_SIZE: int = 200

EPISODES: int = 20000
LOG_INTERVAL: int = 100

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

    for episode in range(1, EPISODES + 1):
        # Simulate trajectory with the current Q-table
        trajectory = run_rollout(env, q_table, explore_factor, rng)
        transitions.insert(trajectory)
        if len(transitions) < BATCH_SIZE:
            continue

        # Sample random batch for training
        train_sample = transitions.sample(BATCH_SIZE)
        for transition in train_sample:
            state_index = transition.state.get_index()
            target = transition.reward
            if not transition.next_state.done:
                target += DISCOUNT_RATE * np.max(q_table[transition.next_state.get_index()])
            # Update the Q-values using the discounted dynamic programming equation
            q_table[state_index][transition.action] += \
                learning_rate * (target - q_table[state_index][transition.action])

        # Decay the explore factor and learning rate
        if explore_factor > MIN_EXPLORE:
            explore_factor = max(explore_factor * EXPLORE_DECAY, MIN_EXPLORE)
        if learning_rate > MIN_LEARNING_RATE:
            learning_rate = max(learning_rate * LEARNING_RATE_DECAY, MIN_LEARNING_RATE)

        if episode % LOG_INTERVAL == 0:
            print(episode, learning_rate, explore_factor, evaluation.evaluate_q_table(q_table))

        if episode % 1000 == 0:
            for row in q_table:
                print([round(float(f), 2) for f in row])

    print(evaluation.evaluate_optimal())
    print(evaluation.evaluate_optimal())
    print(evaluation.evaluate_optimal())