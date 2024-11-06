from numpy.random import Generator
import torch

from env.environment import GamblerState

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
    size: int
    transitions: list[Transition]
    index: int
    rng: Generator

    def __init__(self, size: int, rng: Generator):
        self.size = size
        self.transitions = []
        self.index = 0
        self.rng = rng

    def insert(self, transitions: list[Transition]):
        """Adds transitions to the buffer."""
        for transition in transitions:
            if len(self.transitions) < self.size:
                self.transitions.append(transition)
            else:
                self.transitions[self.index] = transition
                self.index = (self.index + 1) % self.size

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

class TensorSample:
    """A sample from a tensor transition buffer."""
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    next_aciton_masks: torch.Tensor
    done_mask: torch.Tensor

    def __init__(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_action_masks: torch.Tensor,
        done_mask: torch.Tensor,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.next_action_masks = next_action_masks
        self.done_mask = done_mask

class TensorTransitionBuffer:
    """A transition buffer that stores data in tensors for efficient sampling."""
    size: int
    length: int
    index: int
    rng: Generator

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    next_action_masks: torch.Tensor
    done_mask: torch.Tensor

    def insert(self, transitions: list[Transition]):
        """Adds transitions to the buffer."""

    def sample(self, count: int) -> TensorSample:
        """Returns random transitions from the buffer."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Returns the number of transitions in the buffer."""
        return self.length