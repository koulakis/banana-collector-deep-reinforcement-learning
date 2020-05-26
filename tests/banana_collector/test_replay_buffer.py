import pytest

import numpy as np
import pandas as pd


def test_prioritized_replay_buffer_buffers_inputs_with_maximum_priority(dummy_prioritized_replay_buffer):
    frames = [(np.array([i]), 0, 0.0, np.array([]), False) for i in range(20)]

    for frame in frames:
        dummy_prioritized_replay_buffer.add(*frame)

    state_numbers = [int(f[1]['state']) for f in dummy_prioritized_replay_buffer.memory]
    priorities = [int(f[0]) for f in dummy_prioritized_replay_buffer.memory]
    assert state_numbers == [15, 16, 17, 18, 19]
    assert priorities == [1, 1, 1, 1, 1]


class TestSampling:
    @pytest.fixture(autouse=True)
    def setup(
            self,
            dummy_priorities,
            dummy_prioritized_replay_buffer_with_made_up_queue,
            dummy_prioritize_replay_buffer_with_made_up_queue_samples):
        self.buffer = dummy_prioritized_replay_buffer_with_made_up_queue
        self.samples = dummy_prioritize_replay_buffer_with_made_up_queue_samples
        self.alpha = self.buffer.alpha
        self.beta_0 = self.buffer.beta_0
        self.expected_frequency = dummy_priorities ** self.alpha / (dummy_priorities ** self.alpha).sum()
        self.buffer_lth = len(self.buffer)
        self.anneal_steps = self.buffer.beta_number_annealing_steps

    def test_sampling_follows_the_buffered_distribution_changed_by_alpha(self):
        frequencies = (
            pd.DataFrame({
                'state_n': [
                    int(state)
                    for sample in self.samples
                    for state in sample[0]]})
                .groupby('state_n')
                .size()
                .pipe(lambda s: s / s.sum()))

        np.testing.assert_almost_equal(frequencies, self.expected_frequency, decimal=2)

    def test_weights_before_annealing(self):
        sample_weights, sample_idxs = self.samples[0][5:7]

        expected_weights = (self.buffer_lth * self.expected_frequency[sample_idxs]) ** (-self.beta_0)
        expected_weights = expected_weights / expected_weights.max()

        np.testing.assert_almost_equal(sample_weights, expected_weights, decimal=2)

    def test_weights_during_annealing(self):
        sample_weights_annealed, sample_idxs_annealed = self.samples[self.anneal_steps // 2 - 1][5:7]
        expected_weights_annealed = \
            (self.buffer_lth * self.expected_frequency[sample_idxs_annealed]) ** (-(self.beta_0 + 1) / 2.0)
        expected_weights_annealed = expected_weights_annealed / expected_weights_annealed.max()

        np.testing.assert_almost_equal(sample_weights_annealed, expected_weights_annealed, decimal=2)

    def test_weights_after_annealing(self):
        sample_weights_annealed, sample_idxs_annealed = self.samples[self.anneal_steps][5:7]
        expected_weights_annealed = (self.buffer_lth * self.expected_frequency[sample_idxs_annealed]) ** (-1.0)
        expected_weights_annealed = expected_weights_annealed / expected_weights_annealed.max()

        np.testing.assert_almost_equal(sample_weights_annealed, expected_weights_annealed, decimal=2)


def test_prioritized_replay_buffer_updates_priorities(
        dummy_priorities, dummy_prioritized_replay_buffer_with_made_up_queue):
    idxs = [0, 3, 1, 0]
    priorities = [0.3, 2.1, -1.3, 0.8]

    dummy_prioritized_replay_buffer_with_made_up_queue.update_priorities(idxs, priorities)

    buffer_priorities = [dummy_prioritized_replay_buffer_with_made_up_queue.memory[i][0] for i in range(5)]
    assert buffer_priorities == [0.8, 1.3, dummy_priorities[2], 2.1, dummy_priorities[4]]


class TestPrioritizedReplayBufferPerformance:
    def test_buffering_frames_is_fast(self, dummy_real_size_prioritized_replay_buffer):
        pass

    def test_(self, dummy_real_size_prioritized_replay_buffer):
        pass

    def test_sampling_is_almost_constant(self, dummy_real_size_prioritized_replay_buffer):
        pass
