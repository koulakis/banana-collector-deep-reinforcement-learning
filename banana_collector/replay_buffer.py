import random
from collections import namedtuple, deque
from typing import Deque, Any

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
        print('Using new replay buffer.')
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
            tuples: Deque[namedtuple],
            attribute: str,
            cast_bool_to_int: bool = False,
            return_type: Any = torch.FloatTensor
    ) -> torch.Tensor:
        array = np.vstack([e[attribute] for e in tuples if e is not None])
        if cast_bool_to_int:
            array = array.astype(np.uint8)

        return torch.from_numpy(array).type(return_type).to(self.device)
