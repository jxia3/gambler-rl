import numpy as np
from numpy.random import Generator

from environment import GamblerGame
from eval.evaluation import Evaluation
from q_learning.buffer import Transition, TransitionBuffer

# Q-table evaluation parameters
EVAL_EPISODES: int = 10000
EVAL_SEED: int = 1000

# Training parameters
DISCOUNT_RATE: float = 1
LEARNING_RATE: float = 0.005

INITIAL_EXPLORE: float = 1
EXPLORE_DECAY: float = 0.998
MIN_EXPLORE: float = 0.02
BUFFER_SIZE: int = 2000
BATCH_SIZE: int = 100

EPISODES: int = 3000
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

    # Initialize training
    q_table = np.zeros((env.get_state_size(), env.get_action_size()), dtype=np.float32)
    transitions = TransitionBuffer(BUFFER_SIZE, rng)
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
                LEARNING_RATE * (target - q_table[state_index][transition.action])

        # Decay the explore factor
        if explore_factor > MIN_EXPLORE:
            explore_factor = max(explore_factor * EXPLORE_DECAY, MIN_EXPLORE)

        if episode % LOG_INTERVAL == 0:
            print(episode, explore_factor, evaluation.evaluate_q_table(q_table))
            for row in q_table:
                print([round(float(f), 3) for f in row])
            print()

    optimal_table = create_optimal_table(env)
    print(evaluate_q_table(env, optimal_table))