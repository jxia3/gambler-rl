import numpy as np
from numpy.random import Generator

from environment import GamblerGame, GamblerState

# Q-table evaluation parameters
EVAL_EPISODES: int = 10000
EVAL_SEED: int = 1000

# Training parameters
DISCOUNT_RATE: float = 1
LEARNING_RATE: float = 0.005

INITIAL_EXPLORE: float = 1
EXPLORE_DECAY: float = 0.999
MIN_EXPLORE: float = 0.02

EPISODES: int = 10000
LOG_INTERVAL: int = 1000

class Transition:
    """
    A tuple (s_t, a_t, r_t, s_{t + 1}) tracking a state transition in the Markov
    decision process.
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

def evaluate_q_table(train_env: GamblerGame, q_table: np.ndarray):
    """Evaluates the average reward of a Q-table policy by running random episodes."""
    env = GamblerGame(train_env.target_wealth, train_env.win_prob, EVAL_SEED)
    total_reward = 0

    for e in range(EVAL_EPISODES):
        state = env.reset()
        while not state.done:
            values = q_table[state.get_index()].copy()
            values[~state.get_action_mask().numpy()] = -np.inf
            action = int(np.argmax(values))
            reward, state = env.step(state, action)
            total_reward += reward

    return total_reward / EVAL_EPISODES

def create_optimal_table(env: GamblerGame) -> np.ndarray:
    """Creates an optimal Q-table for a gambler game."""
    q_table = np.zeros((env.get_state_size(), env.get_action_size()), dtype=np.float32)
    for wealth in range(1, env.target_wealth):
        bet_amount = 1
        if env.win_prob < 0.5:
            bet_amount = min(wealth, env.target_wealth - wealth)
        q_table[wealth][bet_amount - 1] = 1
    return q_table

def train(env: GamblerGame, seed: int):
    """Trains a tabular Q-learning agent on the gambler Markov decision process."""
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.get_state_size(), env.get_action_size()), dtype=np.float32)
    explore_factor = INITIAL_EXPLORE

    for episode in range(1, EPISODES + 1):
        # Simulate trajectory with the current Q-table network
        trajectory = run_rollout(env, q_table, explore_factor, rng)

        # Update the Q-values for using the discounted dynamic programming equation
        for transition in trajectory:
            target = transition.reward
            if not transition.next_state.done:
                target += DISCOUNT_RATE * np.max(q_table[transition.next_state.get_index()])
            q_table[transition.state.get_index()][transition.action] += \
                LEARNING_RATE * (target - q_table[transition.state.get_index()][transition.action])

        # Decay the explore factor
        if explore_factor > MIN_EXPLORE:
            explore_factor = max(explore_factor * EXPLORE_DECAY, MIN_EXPLORE)

        if episode % LOG_INTERVAL == 0:
            print(explore_factor)
            for row in q_table:
                print(list([round(float(f), 6) for f in row]))
            print(evaluate_q_table(env, q_table))

    optimal_table = create_optimal_table(env)
    print(evaluate_q_table(env, optimal_table))