from collections import deque
from pathlib import Path
from typing import List

import numpy as np
from udacity_custom_unity_agents.unityagents import UnityEnvironment

from banana_collector.agent import Agent


def train_agent(
        agent: Agent,
        env: UnityEnvironment,
        brain_name: str,
        agent_save_path: Path,
        number_of_episodes: int = 2000,
        maximum_timestaps: int = 1000,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.01,
        epsilon_decay: float = 0.995,
        solution_threshold: float = 13
) -> List[float]:
    """Train and save an agent using an environment.

    Args:
        agent: the agent to train
        env: the environment with witch the agent interacts during training
        brain_name: name of the environment brain
        agent_save_path: an output path to save the trained agent parameters
        number_of_episodes: maximum number of training episodes
        maximum_timestaps: maximum number of timesteps per episode
        initial_epsilon: starting value of epsilon, for epsilon-greedy action selection
        final_epsilon: minimum value of epsilon
        epsilon_decay: multiplicative factor (per episode) for decreasing epsilon
        solution_threshold: if this threshold is exceeded from the average of 100 consecutive episodes, then the
            environment is considered solved

    Returns:
        a list of the rolling 100 episode score averages
    """
    scores = []
    solved = False
    scores_window = deque(maxlen=100)
    epsilon = initial_epsilon

    for i_episode in range(1, number_of_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(maximum_timestaps):
            action = agent.act(state, epsilon=epsilon)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        epsilon = max(final_epsilon, epsilon_decay * epsilon)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end='')
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        if np.mean(scores_window) >= solution_threshold and not solved:
            print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            solved = True

    agent.save(agent_save_path)
    return scores
