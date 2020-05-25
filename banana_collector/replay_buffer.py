import random
from collections import deque
from typing import Deque, Any, Dict, List

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int, device: torch.device):
        """Initialize a ReplayBuffer object.

        Args:
            action_size: dimension of each action
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
            device: the device to which samples are loaded
            seed: random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.seed = random.seed(seed)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add a new experience to memory."""
        experience = dict(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.memory.append(experience)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = self._gather_attributes_to_tensor(experiences, 'state')
        actions = self._gather_attributes_to_tensor(experiences, 'action', return_type=torch.LongTensor)
        rewards = self._gather_attributes_to_tensor(experiences, 'reward')
        next_states = self._gather_attributes_to_tensor(experiences, 'next_state')
        dones = self._gather_attributes_to_tensor(experiences, 'done', cast_bool_to_int=True)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def _gather_attributes_to_tensor(
            self,
            tuples: Deque[Dict[str, Any]],
            attribute: str,
            cast_bool_to_int: bool = False,
            return_type: Any = torch.FloatTensor
    ) -> torch.Tensor:
        array = np.vstack([e[attribute] for e in tuples if e is not None])
        if cast_bool_to_int:
            array = array.astype(np.uint8)

        return torch.from_numpy(array).type(return_type).to(self.device)


class PrioritizedReplayBuffer:
    """Fixed size buffer implementing prioritized experience replay"""
    def __init__(
            self,
            action_size: int,
            buffer_size: int,
            batch_size: int,
            seed: int,
            device: torch.device,
            alpha: float = 0.6,
            beta_0: float = 0.4,
            beta_number_annealing_steps=2000):
        """Initialize a ReplayBuffer object.

            Args:
                action_size: dimension of each action
                buffer_size: maximum size of buffer
                batch_size: size of each training batch
                device: the device to which samples are loaded
                seed: random seed
                alpha: prioritization exponent
                beta_0: bias correcting weights exponent - initial value
                beta_number_annealing_steps: number of steps till beta transitions from b_0 to 1 - should be
                    close to the expected number of training updates
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta_0 = beta_0
        self.beta_number_annealing_steps = beta_number_annealing_steps
        self.device = device
        self.seed = random.seed(seed)
        self.sample_number = 0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add a new experience to memory with maximum priority."""
        experience = dict(state=state, action=action, reward=reward, next_state=next_state, done=done)

        maximum_priority = max(map(lambda t: t[0], self.memory)) if any(self.memory) else 1.0
        self.memory.append((maximum_priority, experience))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        self.sample_number += 1
        priorities = np.array([t[0] for t in self.memory]) ** self.alpha
        probabilities = priorities / priorities.sum()

        experiences_idx = np.random.choice(range(len(self.memory)), self.batch_size, p=probabilities)
        experiences = [self.memory[idx][1] for idx in experiences_idx]

        states = self._gather_attributes_to_tensor(experiences, 'state')
        actions = self._gather_attributes_to_tensor(experiences, 'action', return_type=torch.LongTensor)
        rewards = self._gather_attributes_to_tensor(experiences, 'reward')
        next_states = self._gather_attributes_to_tensor(experiences, 'next_state')
        dones = self._gather_attributes_to_tensor(experiences, 'done', cast_bool_to_int=True)

        weights = (len(self.memory) * probabilities[experiences_idx]) ** -self._annealed_beta()
        weights /= weights.max()

        return states, actions, rewards, next_states, dones, weights, experiences_idx

    def update_priorities(self, experiences_idx: List[int], priorities: List[float]):
        """Update the priorities of the elements of the queue belonging to a batch."""
        for idx, priority in zip(experiences_idx, priorities):
            self.memory[idx] = (float(priority), self.memory[idx][1])

    def _gather_attributes_to_tensor(
            self,
            tuples: List[Dict[str, Any]],
            attribute: str,
            cast_bool_to_int: bool = False,
            return_type: Any = torch.FloatTensor
    ) -> torch.Tensor:
        array = np.vstack([e[attribute] for e in tuples if e is not None])
        if cast_bool_to_int:
            array = array.astype(np.uint8)

        return torch.from_numpy(array).type(return_type).to(self.device)

    def _annealed_beta(self) -> float:
        """Return the annealed value of beta. It should range from b_0 to 1."""
        return (
            self.beta_0 * max(0.0, 1.0 - self.sample_number / self.beta_number_annealing_steps)
            + min(1.0, self.sample_number / self.beta_number_annealing_steps))

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
