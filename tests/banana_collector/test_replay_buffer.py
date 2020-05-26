from collections import deque

import numpy as np
import pandas as pd


class TestPrioritizedReplayBufferFunctionality:
    def test_buffers_inputs_with_maximum_priority(self, dummy_prioritized_replay_buffer):
        frames = [(np.array([i]), 0, 0.0, np.array([]), False) for i in range(20)]

        for frame in frames:
            dummy_prioritized_replay_buffer.add(*frame)

        state_numbers = [int(f[1]['state']) for f in dummy_prioritized_replay_buffer.memory]
        priorities = [int(f[0]) for f in dummy_prioritized_replay_buffer.memory]
        assert state_numbers == [15, 16, 17, 18, 19]
        assert priorities == [1, 1, 1, 1, 1]

    def test_sampling_follows_the_buffered_distribution_changed_by_alpha(self, dummy_prioritized_replay_buffer):
        priorities = np.array([1, 0.5, 0.1, 0.3, 1])
        made_up_queue = list(zip(
            priorities,
            [dict(
                state=np.array([i]),
                action=0,
                reward=0.0,
                next_state=np.array([]),
                done=False)
             for i in range(5)]
        ))
        for e in made_up_queue:
            dummy_prioritized_replay_buffer.memory.append(e)

        samples = [dummy_prioritized_replay_buffer.sample() for _ in range(10000)]

        frequencies = (
            pd.DataFrame({
                'state_n': [
                    int(state)
                    for sample in samples
                    for state in sample[0]]})
            .groupby('state_n')
            .size()
            .pipe(lambda s: s / s.sum()))

        alpha = dummy_prioritized_replay_buffer.alpha
        expected_frequency = priorities ** alpha / (priorities ** alpha).sum()
        np.testing.assert_almost_equal(frequencies, expected_frequency, decimal=2)

    def test_weights_are_correctly_computed(self, dummy_prioritized_replay_buffer):
        pass

    def test_updates_priorities(self, dummy_prioritized_replay_buffer):
        pass

    def test_anneals_beta(self, dummy_prioritized_replay_buffer):
        pass

    def test_converges_to_enforced_distribution(self, dummy_prioritized_replay_buffer):
        pass


class TestPrioritizedReplayBufferPerformance:
    def test_buffering_frames_is_fast(self, dummy_real_size_prioritized_replay_buffer):
        pass

    def test_(self, dummy_real_size_prioritized_replay_buffer):
        pass

    def test_sampling_is_almost_constant(self, dummy_real_size_prioritized_replay_buffer):
        pass
