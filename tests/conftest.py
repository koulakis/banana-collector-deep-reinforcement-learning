import pytest
import torch

from banana_collector.replay_buffer import PrioritizedReplayBuffer


@pytest.fixture
def dummy_prioritized_replay_buffer():
    return PrioritizedReplayBuffer(
        action_size=2,
        buffer_size=5,
        batch_size=2,
        seed=None,
        device=torch.device('cpu'),
        beta_number_annealing_steps=20)


@pytest.fixture
def dummy_real_size_prioritized_replay_buffer():
    return PrioritizedReplayBuffer(
        action_size=4,
        buffer_size=int(1e5),
        batch_size=64,
        seed=None,
        device=torch.device('cpu'))
