import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Agent(ABC):
    """Base class for agents with one dimensional state & action inputs."""
    def __init__(self, state_size: int, action_size: int, seed: Optional[int] = None):
        """Initialize an Agent object.

        Args:
            state_size: dimension of each state
            action_size: dimension of each action
            seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

    @abstractmethod
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        pass

    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        pass


class RandomAgent(Agent):
    def __init__(self, state_size: int, action_size: int, seed: Optional[int] = None):
        super().__init__(state_size, action_size, seed)

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """No training for the random agent."""
        pass

    def act(self, state: np.ndarray) -> int:
        """Randomly select and return an action."""
        return random.randint(0, self.action_size)
