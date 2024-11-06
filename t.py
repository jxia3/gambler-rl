from q_learning.buffer import TensorTransitionBuffer
import numpy as np

rng = np.random.default_rng()
buffer = TensorTransitionBuffer(5, rng)