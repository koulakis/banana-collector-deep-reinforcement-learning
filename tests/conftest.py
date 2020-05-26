from typing import List, Tuple
from collections import deque

import pytest
import torch
import numpy as np

from banana_collector.replay_buffer import PrioritizedReplayBuffer


def _generate_prioritized_replay_buffer() -> PrioritizedReplayBuffer:
    return PrioritizedReplayBuffer(
        action_size=2,
        buffer_size=5,
        batch_size=2,
        seed=None,
        device=torch.device('cpu'),
        beta_number_annealing_steps=20)


@pytest.fixture
def dummy_prioritized_replay_buffer():
    return _generate_prioritized_replay_buffer()


@pytest.fixture(scope='class')
def dummy_priorities():
    return np.array([1, 0.5, 0.1, 0.3, 1])


@pytest.fixture(scope='class')
def dummy_prioritized_replay_buffer_with_made_up_queue(dummy_priorities) -> PrioritizedReplayBuffer:
    made_up_queue = list(zip(
        dummy_priorities,
        [dict(
            state=np.array([i]),
            action=0,
            reward=0.0,
            next_state=np.array([]),
            done=False)
            for i in range(5)]
    ))
    buffer = _generate_prioritized_replay_buffer()
    for e in made_up_queue:
        buffer.memory.append(e)
    return buffer


@pytest.fixture(scope='class')
def dummy_prioritize_replay_buffer_with_made_up_queue_samples(
        dummy_prioritized_replay_buffer_with_made_up_queue
) -> List[Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[int]]]:
    return [dummy_prioritized_replay_buffer_with_made_up_queue.sample() for _ in range(10000)]


@pytest.fixture(scope='class')
def dummy_real_size_prioritized_replay_buffer():
    buffer = PrioritizedReplayBuffer(
        action_size=4,
        buffer_size=int(1e5),
        batch_size=64,
        seed=None,
        device=torch.device('cpu'))

    buffer.memory.extend(zip(
        range(int(1e5)),
        [dict(
            state=np.array([i]),
            action=0,
            reward=0.0,
            next_state=np.array([]),
            done=False)
            for i in range(int(1e5))]))

    return buffer
