import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch import optim

from banana_collector.network import FullyConnected

DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
    """Agent which executes random actions. It is a dummy agent to be used for testing."""
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """No training for the random agent."""
        pass

    def act(self, state: np.ndarray) -> int:
        """Randomly select and return an action."""
        return random.randint(0, self.action_size)


class DqnAgent(Agent):
    """An agent using a DQN to approximate a value function."""
    def __init__(
            self,
            state_size: int,
            action_size: int,
            learning_rate: float = 1e-4,
            device: str = DEFAULT_DEVICE,
            seed: Optional[int] = None):
        super().__init__(state_size, action_size)
        self.seed = random.seed(seed)
        self.device = device

        self.local_dqn = FullyConnected(state_size=state_size, action_size=action_size).to(device)
        self.target_dqn = FullyConnected(state_size=state_size, action_size=action_size).to(device)

        self.optimizer = optim.Adam(self.local_dqn.parameters(), lr=learning_rate)

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Gather experience for a single step. The information is saved on a replay buffer and training is
        triggered with a fixed frequency."""
        pass

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """For a given input state compute the value of the Q-function using on the local network. A parameter
        epsilon can be set to apply epsilon-greedy selection during training."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_dqn.eval()
        with torch.no_grad():
            action_values = self.local_dqn(state)
        self.local_dqn.train()

        if random.random() > epsilon:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))
