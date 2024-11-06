from q_learning.buffer import Transition, TensorTransitionBuffer
import numpy as np
from env.environment import GamblerGame

env = GamblerGame(6, 0.4, 1)
state = env.reset()
rng = np.random.default_rng(24311)
buffer = TensorTransitionBuffer(5, env.get_state_size(), env.get_action_size(), rng)

print(state)
r1, s2 = env.step(state, 1)
print(r1, s2)
r2, s3 = env.step(s2, 4)
print(r2, s3)

t = [Transition(state, 1, r1, s2), Transition(s2, 4, r2, s3)]
buffer.insert(t)
buffer.insert(t)
buffer.insert(t)

print()
print(buffer.size, buffer.length, buffer.index)
print(buffer.observations)
print(buffer.actions)
print(buffer.rewards)
print(buffer.next_observations)
print(buffer.next_action_masks)
print(buffer.done_mask)

sample = buffer.sample(2)
print()
print()
print(sample.observations)
print(sample.actions)
print(sample.rewards)
print(sample.next_observations)
print(sample.next_action_masks)
print(sample.done_mask)