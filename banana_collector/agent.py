import random
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from pathlib import Path

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch import nn

from banana_collector.network import FullyConnectedNetwork, DuelingNetwork
from banana_collector.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

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
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        pass

    @abstractmethod
    def save(self, output_path: Path):
        pass

    @abstractmethod
    def load(self, input_path: Path):
        pass


class RandomAgent(Agent):
    """Agent which executes random actions. It is a dummy agent to be used for testing."""
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """No training for the random agent."""
        pass

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Randomly select and return an action."""
        return random.randint(0, self.action_size)

    def save(self, output_path: Path):
        pass

    def load(self, input_path: Path):
        pass


class DqnAgent(Agent):
    """An agent using a DQN to approximate a value function."""
    def __init__(
            self,
            state_size: int,
            action_size: int,
            learning_rate: float = 5e-4,
            device: torch.device = DEFAULT_DEVICE,
            buffer_size: int = int(1e5),
            batch_size: int = 64,
            update_every_global_step: int = 4,
            gamma: float = 0.99,
            tau: float = 1e-3,
            hidden_layers: Optional[List[int]] = None,
            double_dqn: bool = True,
            prioritize_replay: bool = False,
            dueling_dqn: bool = True,
            per_alpha: Optional[float] = None,
            per_beta_0: Optional[float] = None,
            seed: Optional[int] = None):
        super().__init__(state_size, action_size)
        self.seed = random.seed(seed)
        self.device = device
        self.prioritize_replay = prioritize_replay

        self.local_dqn = (DuelingNetwork if dueling_dqn else FullyConnectedNetwork)(
            input_size=state_size,
            output_size=action_size,
            hidden_layer_widths=hidden_layers).to(device)
        self.target_dqn = (DuelingNetwork if dueling_dqn else FullyConnectedNetwork)(
            input_size=state_size,
            output_size=action_size,
            hidden_layer_widths=hidden_layers).to(device)

        print(self.local_dqn)

        self.optimizer = optim.Adam(self.local_dqn.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            'max',
            factor=0.5,
            cooldown=100,
            patience=200,
            min_lr=1e-7,
            verbose=True)
        self.memory = (PrioritizedReplayBuffer if self.prioritize_replay else ReplayBuffer)(
            action_size=action_size,
            buffer_size=buffer_size,
            batch_size=batch_size,
            seed=seed,
            device=device,
            **(dict(
                alpha=per_alpha,
                beta_0=per_beta_0)
                if self.prioritize_replay
               else dict()))
        self.global_step = 0
        self.update_every_global_step = update_every_global_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.double_dqn = double_dqn

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Gather experience for a single step. The information is saved on a replay buffer and training is
        triggered with a fixed frequency."""
        self.memory.add(state, action, reward, next_state, done)

        self.global_step += 1
        if self.global_step % self.update_every_global_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

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

    def save(self, output_path: Path):
        """Save the weights of the local network."""
        torch.save(self.local_dqn.state_dict(), output_path)

    def load(self, input_path: Path):
        """Load the weights of the local network."""
        self.local_dqn.load_state_dict(torch.load(input_path))

    def learn(self, experiences: List[torch.Tensor]):
        """Update value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences[:5]

        argmax_dqn = self.local_dqn if self.double_dqn else self.target_dqn
        argmax_dqn.eval()
        self.target_dqn.eval()
        with torch.no_grad():
            argmax_values = argmax_dqn(next_states).argmax(dim=1, keepdim=True)
            target_max_values = self.target_dqn(next_states).gather(1, argmax_values).detach()
        argmax_dqn.train()
        self.target_dqn.train()

        expected_rewards = rewards + (1.0 - dones) * self.gamma * target_max_values
        predicted_rewards = self.local_dqn(states).gather(1, actions)

        if self.prioritize_replay:
            weights, experiences_idx = experiences[5:7]
            td_error = expected_rewards - predicted_rewards
            loss = (weights * F.mse_loss(expected_rewards, predicted_rewards, reduction='none')).mean()

            self.memory.update_priorities(experiences_idx, np.abs(td_error.detach().cpu().numpy()))

        else:
            loss = F.mse_loss(expected_rewards, predicted_rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local_dqn, self.target_dqn, self.tau)

    def on_epoch_end(self, average_score):
        self.scheduler.step(average_score)

    @staticmethod
    def soft_update(local_model: nn.Module, target_model: nn.Module, tau: float):
        """Update the target model parameters with a fraction of the local model parameters regulated by tau:
        θ_target = τ*θ_local + (1 - τ)*θ_target.

        Args:
            local_model: weights will be copied from
            target_model: weights will be copied to
            tau: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
