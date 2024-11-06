from q_learning.buffer import TensorTransitionBuffer
import numpy as np
from env.environment import GamblerGame

env = GamblerGame(6, 0.4, 0)
state = env.reset()
rng = np.random.default_rng()
buffer = TensorTransitionBuffer(5, env.get_state_size(), env.get_action_size(), rng)

print(buffer.observations)
print(buffer.actions)
print(buffer.rewards)
print(buffer.next_observations)
print(buffer.next_action_masks)
print(buffer.done_mask)